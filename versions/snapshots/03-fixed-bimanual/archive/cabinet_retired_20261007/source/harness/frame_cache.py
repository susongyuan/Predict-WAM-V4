"""Reader for packed-JPEG frame caches produced by scripts/cache_behavior_frames.py."""

import json
from pathlib import Path

import numpy as np


class FramePack:
    """One episode/camera: random access to cached frames by original frame index."""

    def __init__(self, directory, episode_id):
        directory = Path(directory)
        self.path = directory / f"episode_{episode_id}.jpgpack"
        self.meta = json.loads(self.path.with_suffix(".meta.json").read_text())
        if not self.meta.get("complete"):
            raise ValueError(f"Incomplete frame cache: {self.path}")
        self.offsets = np.load(self.path.with_suffix(".idx.npy"))
        self.stride = int(self.meta["stride"])
        self.length = int(self.meta["length"])
        self.count = int(self.meta["count"])
        if len(self.offsets) != self.count + 1:
            raise ValueError("Offset table does not match frame count")
        self._mm = None

    def _map(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def has(self, frame_index):
        return 0 <= frame_index < self.length and frame_index % self.stride == 0

    def jpeg(self, frame_index):
        if not self.has(frame_index):
            raise KeyError(f"Frame {frame_index} not cached (stride {self.stride}, length {self.length})")
        slot = frame_index // self.stride
        start, end = int(self.offsets[slot]), int(self.offsets[slot + 1])
        return bytes(self._map()[start:end])

    def rgb(self, frame_index):
        """Decode to HxWx3 uint8 RGB with OpenCV (BGR->RGB conversion applied)."""
        import cv2
        array = cv2.imdecode(np.frombuffer(self.jpeg(frame_index), dtype=np.uint8), cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError("JPEG decode failed")
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
