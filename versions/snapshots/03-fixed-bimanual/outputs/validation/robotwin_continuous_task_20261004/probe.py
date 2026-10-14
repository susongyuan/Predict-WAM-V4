"""Read a bounded official ZIP directory and one episode, without downloading the archive."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import h5py

OUT = Path(__file__).resolve().parent
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
URL = ("https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/"
       + REVISION + "/dataset/put_object_cabinet/aloha-agilex_randomized_500.zip")
SIZE = 11014603365


class RemoteRange(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.transferred = 0
        self.requests = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: SIZE}[whence]
        self.position = base + offset
        if not 0 <= self.position <= SIZE:
            raise ValueError("Seek outside official archive")
        return self.position

    def read(self, length=-1):
        if length < 0:
            length = SIZE - self.position
        length = min(length, SIZE - self.position)
        if not length:
            return b""
        if length > 64 * 1024**2 or self.transferred + length > 96 * 1024**2:
            raise ValueError("Probe exceeds bounded network budget")
        start, end = self.position, self.position + length - 1
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = f"bytes {start}-{end}/{SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError(f"Server did not honor exact Range: {response.status}, {response.headers.get('Content-Range')}")
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("Partial or oversized range response")
        self.requests.append({"start": start, "end": end, "bytes": length})
        self.transferred += length
        self.position += length
        return data


report = {"url": URL, "revision": REVISION, "archive_bytes": SIZE,
          "full_archive_sha256_verified": False, "status": "running", "training": False}
try:
    remote = RemoteRange()
    with zipfile.ZipFile(remote) as bundle:
        entries = bundle.infolist()
        episodes = sorted((x for x in entries if x.filename.endswith((".h5", ".hdf5"))), key=lambda x: x.filename)
        report.update(archive_entries=len(entries), hdf5_members=len(episodes),
                      extracted_total_bytes=sum(x.file_size for x in entries),
                      episode_member_names=[x.filename for x in episodes])
        if not episodes or episodes[0].file_size > 64 * 1024**2:
            raise ValueError("No bounded single episode")
        first = episodes[0]
        blob = bundle.read(first)
        destination = OUT / "sample_episode.hdf5"
        with destination.open("xb") as f:
            f.write(blob)
        report.update(sample_member=first.filename, sample_bytes=len(blob),
                      sample_sha256=hashlib.sha256(blob).hexdigest(),
                      sample_crc32_verified=True, network_bytes=remote.transferred,
                      range_requests=remote.requests)
    shapes = {}
    with h5py.File(destination, "r") as file:
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                shapes[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        file.visititems(visit)
        report["dataset_schema"] = shapes
        report["root_attributes"] = {str(k): str(v) for k, v in file.attrs.items()}
    report["status"] = "single_episode_format_probed"
except Exception as error:
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    (OUT / "archive_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episode_member_names"}, ensure_ascii=False, indent=2))
