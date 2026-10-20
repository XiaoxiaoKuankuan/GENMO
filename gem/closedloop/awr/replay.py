"""完整Episode FIFO回放、不可变快照与引用保护。

物理rank各自发布二进制Episode目录，唯一主写入者用SQLite事务登记。不同Actor
版本可并存，但奖励/归一化/GMT/物理契约必须一致。容量按完整Episode计算，
淘汰不会截断回报链；被checkpoint或对照快照引用的数据在释放引用前不删除。
元数据和数组分开存储，重复登记幂等，坏校验/验证集泄漏/磁盘预算不足均拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import time

import torch
from .io import atomic_json, digest, read_tree, write_tree
from .records import AWRCompleteEpisode
from .returns import mc_returns


@dataclass(frozen=True)
class ReplaySnapshot:
    snapshot_id: str
    episode_ids: tuple[str, ...]
    contract_sha256: str
    transition_count: int
    actor_sample_count: int

    def to_dict(self):
        return dict(snapshot_id=self.snapshot_id, episode_ids=list(self.episode_ids),
                    contract_sha256=self.contract_sha256, transition_count=self.transition_count,
                    actor_sample_count=self.actor_sample_count)


class EpisodeReplay:
    def __init__(self, root, *, capacity, contract, heldout_groups=(), heldout_audio=(),
                 min_free_bytes=0, max_bytes=500 * 1024**3, read_only_index=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity, self.contract = int(capacity), contract
        if self.capacity < 1:
            raise ValueError("Replay capacity is in complete episodes")
        self.contract_sha = digest(contract)
        self.heldout_groups, self.heldout_audio = set(heldout_groups), set(heldout_audio)
        self.min_free_bytes, self.max_bytes = min_free_bytes, max_bytes
        self.read_only_index = read_only_index
        self.db = (sqlite3.connect("file:"+str(self.root/"index.sqlite")+"?mode=ro", uri=True)
                   if read_only_index else sqlite3.connect(self.root/"index.sqlite"))
        if not read_only_index:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(seq INTEGER PRIMARY KEY AUTOINCREMENT,
              id TEXT UNIQUE NOT NULL, sha TEXT NOT NULL, source TEXT NOT NULL,
              group_id TEXT NOT NULL, audio TEXT NOT NULL, behavior TEXT NOT NULL,
              transitions INTEGER NOT NULL, samples INTEGER NOT NULL,
              bytes INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS pins(snapshot TEXT NOT NULL,episode TEXT NOT NULL,
              PRIMARY KEY(snapshot,episode));
        """)
        settings = dict(capacity=capacity, contract_sha=self.contract_sha,
                        holdouts_sha=digest([sorted(self.heldout_groups), sorted(self.heldout_audio)]))
        previous = self.db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
        if previous and json.loads(previous[0]) != settings:
            raise ValueError("Replay contract/capacity/holdout identity changed")
        if not read_only_index:
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO settings VALUES('identity',?)",
                                (json.dumps(settings, sort_keys=True),))
        self.metrics = dict(read_seconds=0., read_bytes=0, read_episodes=0)

    def _validate(self, episode):
        episode.validate()
        if episode.group_id in self.heldout_groups or episode.audio_sha256 in self.heldout_audio:
            raise ValueError("Validation/test music leaked into replay")
        if digest(episode.contract) != self.contract_sha:
            raise ValueError("Replay reward/physics/normalizer identity changed")
        supplied = episode.returns.clone() if episode.returns is not None else None
        expected = mc_returns(episode)
        if supplied is not None and not torch.equal(supplied, expected):
            raise ValueError("Stored MC labels differ from actual rewards")
        return expected

    def publish(self, episode):
        self._validate(episode)
        if shutil.disk_usage(self.root).free < self.min_free_bytes:
            raise OSError("Replay disk reserve exhausted")
        path = self.root / "episodes" / episode.episode_id
        sha = write_tree(path, episode.to_dict())
        return dict(episode_id=episode.episode_id, sha256=sha)

    def register(self, publication):
        identifier = publication["episode_id"]
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid episode content identity")
        path = self.root / "episodes" / identifier
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["sha256"] != publication["sha256"]:
            raise ValueError("Published episode identity mismatch")
        episode = AWRCompleteEpisode.from_dict(read_tree(path))
        supplied = episode.returns.clone() if episode.returns is not None else None
        expected = self._validate(episode)
        if supplied is None or not torch.equal(supplied, expected):
            raise ValueError("Stored MC labels differ from actual rewards")
        old = self.db.execute("SELECT sha FROM episodes WHERE id=?", (identifier,)).fetchone()
        if old:
            if old[0] != publication["sha256"]:
                raise ValueError("Duplicate episode with changed content")
            return False
        size = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
        if self.size_bytes() + size > self.max_bytes:
            raise OSError("Replay byte budget exhausted; release unused snapshots first")
        with self.db:
            self.db.execute("""INSERT INTO episodes
                (id,sha,source,group_id,audio,behavior,transitions,samples,bytes)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (identifier, publication["sha256"], episode.source, episode.group_id,
                 episode.audio_sha256, str(episode.rows[0].identity["policy_version"]),
                 len(episode.rows), sum(r.actor_valid for r in episode.rows), size))
            active = self.db.execute("SELECT id FROM episodes WHERE active=1 ORDER BY seq DESC").fetchall()
            for (expired,) in active[self.capacity:]:
                self.db.execute("UPDATE episodes SET active=0 WHERE id=?", (expired,))
        return True

    def append(self, episode):
        return self.register(self.publish(episode))

    def snapshot(self):
        rows = self.db.execute("SELECT id,sha,transitions,samples FROM episodes WHERE active=1 ORDER BY seq").fetchall()
        if not rows:
            raise ValueError("Cannot label an empty replay")
        identifier = digest(dict(contract=self.contract_sha, episodes=[r[:2] for r in rows]))
        snapshot = ReplaySnapshot(identifier, tuple(r[0] for r in rows), self.contract_sha,
                                  sum(r[2] for r in rows), sum(r[3] for r in rows))
        atomic_json(self.root / "snapshots" / (identifier + ".json"), snapshot.to_dict())
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO pins VALUES(?,?)",
                                [(identifier, r[0]) for r in rows])
        return snapshot

    def load_snapshot(self, identifier):
        value = json.loads((self.root / "snapshots" / (identifier + ".json")).read_text())
        value["episode_ids"] = tuple(value["episode_ids"])
        snapshot = ReplaySnapshot(**value)
        if snapshot.contract_sha256 != self.contract_sha or snapshot.snapshot_id != identifier:
            raise ValueError("Snapshot contract mismatch")
        for episode_id in snapshot.episode_ids:
            if not (self.root / "episodes" / episode_id / "manifest.json").is_file():
                raise ValueError("Snapshot data no longer retained")
        rows = [self.db.execute("SELECT id,sha,transitions,samples FROM episodes WHERE id=?", (key,)).fetchone()
                for key in snapshot.episode_ids]
        if (any(row is None for row in rows)
                or digest(dict(contract=self.contract_sha, episodes=[r[:2] for r in rows])) != identifier
                or sum(r[2] for r in rows) != snapshot.transition_count
                or sum(r[3] for r in rows) != snapshot.actor_sample_count):
            raise ValueError("Snapshot membership/counts no longer match immutable index")
        return snapshot

    def read(self, episode_id):
        started = time.perf_counter()
        record = self.db.execute("SELECT sha,bytes FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if record is None:
            raise KeyError(episode_id)
        path = self.root / "episodes" / episode_id
        if json.loads((path / "manifest.json").read_text())["sha256"] != record[0]:
            raise ValueError("Replay index/shard identity mismatch")
        episode = AWRCompleteEpisode.from_dict(read_tree(path))
        self.metrics["read_seconds"] += time.perf_counter() - started
        self.metrics["read_bytes"] += record[1]
        self.metrics["read_episodes"] += 1
        return episode

    def release(self, snapshot_id):
        with self.db:
            self.db.execute("DELETE FROM pins WHERE snapshot=?", (snapshot_id,))

    def collect_garbage(self):
        rows = self.db.execute("""SELECT id FROM episodes WHERE active=0 AND bytes>0
          AND NOT EXISTS(SELECT 1 FROM pins WHERE pins.episode=episodes.id)""").fetchall()
        for (identifier,) in rows:
            path = (self.root / "episodes" / identifier).resolve()
            if path.parent != self.root / "episodes":
                raise ValueError("Refuse unsafe replay cleanup")
            if path.exists():
                shutil.rmtree(path)
            with self.db:
                self.db.execute("UPDATE episodes SET bytes=0 WHERE id=?", (identifier,))
        return [r[0] for r in rows]

    def size_bytes(self):
        return self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM episodes").fetchone()[0]

    def report(self):
        return dict(self.metrics, bytes=self.size_bytes(), capacity_episodes=self.capacity,
                    active_episodes=self.db.execute("SELECT COUNT(*) FROM episodes WHERE active=1").fetchone()[0])

    def close(self):
        self.db.close()
