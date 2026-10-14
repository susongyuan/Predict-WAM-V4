#!/usr/bin/env python3
"""Fetch the three pinned official archives, verify SHA-256, and atomically expand them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import urllib.request
import zipfile

from behavior_preflight import CONFIG, ROOT


def download_ranges(asset, archive_dir):
    """Four bounded HTTP streams; resume completed and partially completed 1 GiB ranges."""
    target = archive_dir / asset["filename"]
    if target.exists() and target.stat().st_size == asset["bytes"]:
        return target
    parts_dir = archive_dir / (asset["filename"] + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{CONFIG['asset_repository']}/resolve/"
           f"{CONFIG['asset_revision']}/{asset['filename']}")
    size = 1024**3

    def fetch(index):
        path = parts_dir / f"{index:04d}"
        start = index * size
        end = min(start + size, asset["bytes"]) - 1
        for attempt in range(3):
            current = path.stat().st_size if path.exists() else 0
            if current == end - start + 1:
                return path
            if current > end - start + 1:
                raise ValueError(f"Oversized partial range: {path}")
            request = urllib.request.Request(url, headers={"Range": f"bytes={start+current}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, path.open("ab") as handle:
                    expected = f"bytes {start+current}-{end}/{asset['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact requested byte range")
                    for block in iter(lambda: response.read(1024**2), b""):
                        handle.write(block)
                if path.stat().st_size != end - start + 1:
                    raise OSError("Incomplete range response")
                print(f"Range {index+1}/{(asset['bytes']+size-1)//size} complete", flush=True)
                return path
            except (OSError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, range((asset["bytes"] + size - 1) // size)))
    temporary = target.with_suffix(target.suffix + ".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=8 * 1024**2)
    temporary.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--http-parallel", action="store_true", help="Use four bounded HTTP streams instead of Xet")
    args = parser.parse_args()
    if not args.download:
        print(json.dumps(CONFIG["assets"], indent=2))
        return
    if not CONFIG["authorization"]["behavior_asset_license"]:
        raise SystemExit("The user must accept the BEHAVIOR asset license first")
    if not args.http_parallel:
        from huggingface_hub import hf_hub_download
    archive_dir = ROOT / "data/behavior-assets-download"
    archive_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / CONFIG["data_path"]
    data_dir.mkdir(parents=True, exist_ok=True)
    for asset in sorted(CONFIG["assets"], key=lambda item: item["bytes"]):
        marker = data_dir / asset["directory"] / "DOWNLOAD_PROVENANCE.json"
        if marker.exists() and json.loads(marker.read_text()).get("sha256") == asset["sha256"]:
            print(f"Already verified and expanded: {asset['directory']}", flush=True)
            continue
        print(f"Downloading {asset['filename']} ({asset['bytes']} bytes)", flush=True)
        archive = (download_ranges(asset, archive_dir) if args.http_parallel else
                   Path(hf_hub_download(repo_id=CONFIG["asset_repository"], filename=asset["filename"],
                        repo_type="dataset", revision=CONFIG["asset_revision"], local_dir=archive_dir, token=False)))
        if archive.stat().st_size != asset["bytes"]:
            raise RuntimeError(f"Size mismatch: {archive}")
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != asset["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {archive}")
        print(f"SHA-256 verified: {asset['filename']}", flush=True)
        parts_dir = archive_dir / (asset["filename"] + ".parts")
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        target = data_dir / asset["directory"]
        if target.exists():
            raise RuntimeError(f"Refusing to replace existing unverified directory: {target}")
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
            unpacked_bytes = sum(entry.file_size for entry in entries)
            if shutil.disk_usage(data_dir).free < unpacked_bytes + 10 * 1024**3:
                raise RuntimeError("Insufficient disk space for archive expansion plus 10 GiB reserve")
            staging = Path(tempfile.mkdtemp(prefix=asset["directory"] + ".partial-", dir=data_dir))
            try:
                source.extractall(staging)
                provenance = {**asset, "revision": CONFIG["asset_revision"], "unpacked_bytes": unpacked_bytes,
                              "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                (staging / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
                staging.rename(target)
            except BaseException:
                shutil.rmtree(staging)
                raise
        print(f"Expanded {target} ({unpacked_bytes} uncompressed bytes)", flush=True)


if __name__ == "__main__":
    main()
