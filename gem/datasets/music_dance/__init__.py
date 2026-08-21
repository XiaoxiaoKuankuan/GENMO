"""Manifest-backed human and robot music-dance datasets."""

from .music_dance_bumi import BumiMusicDanceDataset
from .music_dance_smpl import MusicDanceSmplDataset

__all__ = ["BumiMusicDanceDataset", "MusicDanceSmplDataset"]
