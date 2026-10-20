"""训练进程内有界的不可变资产字节缓存。

音乐、配对动作、清单和元数据在首次读取时计算完整SHA，后续只有设备号、inode、
大小、纳秒mtime和ctime均一致时复用已验证字节。读取前后再次stat，拒绝读取期间
被替换的文件。缓存不保存可修改的Tensor/NumPy对象，也不参与预算或checkpoint等
可变恢复文件；容量按真实字节限制，超大单文件直接读取但不驻留。clear接口用于
周期完整核验，统计明确区分完整读取与stat复用，不把stat检查称为重新计算SHA。
"""
from collections import OrderedDict
import hashlib
from pathlib import Path


def file_identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class ImmutableBytesCache:
    def __init__(self, max_bytes=128*1024**2):
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = self.reads = self.hits = 0

    def read(self, path):
        path = Path(path).resolve(strict=True)
        before = file_identity(path)
        old = self.entries.pop(path, None)
        if old is not None:
            self.bytes -= len(old[1])
            if old[0] == before:
                self.entries[path] = old
                self.bytes += len(old[1])
                self.hits += 1
                return old[1], old[2]
        data = path.read_bytes()
        if file_identity(path) != before:
            raise RuntimeError(f'Immutable asset changed during read: {path}')
        digest = hashlib.sha256(data).hexdigest()
        self.reads += 1
        if len(data) <= self.max_bytes:
            while self.bytes+len(data) > self.max_bytes:
                _, removed = self.entries.popitem(last=False)
                self.bytes -= len(removed[1])
            self.entries[path] = (before, data, digest)
            self.bytes += len(data)
        return data, digest

    def clear(self):
        self.entries.clear()
        self.bytes = 0


ASSET_BYTES = ImmutableBytesCache()
