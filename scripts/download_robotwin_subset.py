#!/usr/bin/env python3
"""Download one pinned RoboTwin dataset archive, verify it, and extract safely.

No dependency installation, simulation, conversion, or training is performed.
The output must be new; provenance survives removal of the transport archive.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "TianxingChen/RoboTwin2.0"
REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"


def read_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "robotics-subset-downloader/1"}), timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_cans_plasticbox")
    parser.add_argument("--archive", default="demo_clean.zip", choices=("demo_clean.zip", "aloha-agilex_randomized_500.zip"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.task or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in args.task):
        parser.error("task must be a lowercase identifier")
    out = args.out or ROOT / "data/benchmarks/robotwin2" / args.task
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / "SOURCE.json"
    started = time.time()
    record = {"schema_version": 1, "task": args.task, "repo": REPO, "revision": REVISION,
              "archive": args.archive, "status": "metadata_pending", "started_unix": started,
              "scope": "downloaded_demonstrations_not_wam_performance", "gpu_jobs": 0}

    def save():
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        remote = f"dataset/{args.task}/{args.archive}"
        api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/dataset/{args.task}"
        with read_url(api) as response:
            entries = json.load(response)
        entry = next(row for row in entries if row.get("path") == remote)
        expected_size = int(entry["size"])
        lfs = entry.get("lfs", {})
        expected_sha = lfs.get("oid") or lfs.get("sha256")
        if not expected_sha or len(expected_sha) != 64:
            raise ValueError("Missing upstream SHA256; refusing unverifiable dataset download")
        download_limit = (24 if args.archive.endswith("_500.zip") else 4) * 1024**3
        if not 0 < expected_size <= download_limit:
            raise ValueError("Archive outside the registered single-task download budget")
        if shutil.disk_usage(out).free < 3 * expected_size + 2 * 1024**3:
            raise ValueError("Insufficient disk headroom")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}"
        record.update(status="downloading", url=url, upstream_file=entry,
                      expected_bytes=expected_size, expected_sha256=expected_sha)
        save()
        archive = out / (args.archive + ".part")
        digest = hashlib.sha256()
        received = 0
        with read_url(url) as source, archive.open("xb") as target:
            while block := source.read(4 * 1024**2):
                received += len(block)
                if received > expected_size:
                    raise ValueError("Received more bytes than the upstream manifest")
                target.write(block)
                digest.update(block)
        actual_sha = digest.hexdigest()
        if received != expected_size or actual_sha != expected_sha:
            raise ValueError(f"Download mismatch: {received} bytes, sha={actual_sha}")
        record.update(status="archive_verified", received_bytes=received, sha256=actual_sha)
        save()
        destination = out / "source"
        members = []
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            total = sum(info.file_size for info in infos)
            extraction_limit = (40 if args.archive.endswith("_500.zip") else 8) * 1024**3
            if total > extraction_limit or total + 1024**3 > shutil.disk_usage(out).free:
                raise ValueError("Unexpected extracted size or insufficient disk headroom")
            names = set()
            for info in infos:
                path = PurePosixPath(info.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                        or stat.S_ISLNK(info.external_attr >> 16) or info.filename in names):
                    raise ValueError(f"Unsafe or duplicate archive member: {info.filename}")
                names.add(info.filename)
            destination.mkdir(exist_ok=False)
            for info in infos:
                path = destination.joinpath(*PurePosixPath(info.filename).parts)
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024**2)
                if path.stat().st_size != info.file_size:
                    raise ValueError(f"Extraction size mismatch: {info.filename}")
                members.append({"path": str(path.relative_to(out)), "bytes": info.file_size,
                                "zip_crc32": f"{info.CRC:08x}"})
        record.update(status="complete", extracted_bytes=total, members=members,
                      hdf5_files=sum(Path(row["path"]).suffix in (".hdf5", ".h5") for row in members),
                      elapsed_seconds=time.time() - started, zip_crc_verified=True,
                      transport_archive_removed=False)
        save()
        archive.unlink()
        record["transport_archive_removed"] = True
        save()
        print(json.dumps({key: value for key, value in record.items() if key != "members"}, ensure_ascii=False, indent=2))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time() - started)
        save()
        raise


if __name__ == "__main__":
    main()
