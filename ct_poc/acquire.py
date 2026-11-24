"""Fetch selected MSD members with bounded HTTP range reads."""
from pathlib import Path
import hashlib
import json
import os
import re
import tarfile
import tempfile
import requests

ARCHIVE = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def range_read(session, start, size, etag):
    with session.get(ARCHIVE, headers={"Range": f"bytes={start}-{start+size-1}",
                                      "If-Match": etag}, timeout=(15, 90), stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Server ignored the range request; refusing a full archive download.")
        if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{start+size-1}/"):
            raise RuntimeError("Unexpected Content-Range.")
        data = response.raw.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Incomplete archive member.")
        return data

def catalog(root, refresh=False):
    path = Path(root) / "raw" / "catalog.json"
    with requests.Session() as session:
        response = session.head(ARCHIVE, timeout=(15, 30))
        response.raise_for_status()
        etag = response.headers["ETag"]
        total = int(response.headers["Content-Length"])
        if path.exists() and not refresh:
            cached = json.loads(path.read_text())
            if cached.get("etag") == etag and cached.get("url") == ARCHIVE:
                return cached
        members, metadata, position = {}, None, 0
        for _ in range(600):
            header = range_read(session, position, 512, etag)
            if not header.strip(b"\0"):
                break
            item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            offset = position + 512
            if item.size < 0 or offset + item.size > total:
                raise RuntimeError("Invalid tar member size.")
            if re.fullmatch(r"Task09_Spleen/(imagesTr|labelsTr)/spleen_[0-9]+\.nii\.gz", item.name):
                members[item.name.split("/", 1)[1]] = {"offset": offset, "size": item.size}
            elif item.name == "Task09_Spleen/dataset.json" and item.size < 100000:
                metadata = json.loads(range_read(session, offset, item.size, etag))
            position = offset + ((item.size + 511) // 512) * 512
        else:
            raise RuntimeError("Archive member limit exceeded.")
        if metadata is None or metadata.get("labels") != {"0": "background", "1": "spleen"}:
            raise RuntimeError("Unexpected dataset metadata.")
        result = {"url": ARCHIVE, "etag": etag, "archive_bytes": total,
                  "metadata": metadata, "members": members}
        write_json(path, result)
        return result

def training_cases(index):
    names = [Path(pair["image"]).name[:-7] for pair in index["metadata"]["training"]]
    return sorted(names, key=lambda n: int(n.split("_")[-1]))

def fetch_cases(root, cases):
    root = Path(root)
    index = catalog(root)
    manifest_path = root / "raw" / "downloads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    requested = training_cases(index) if cases == ["all"] else cases
    allowed = set(training_cases(index))
    with requests.Session() as session:
        for case_id in requested:
            if case_id not in allowed:
                raise ValueError(f"Unknown labelled training case: {case_id}")
            for kind in ("imagesTr", "labelsTr"):
                name = f"{kind}/{case_id}.nii.gz"
                info = index["members"][name]
                destination = root / "raw" / name
                if destination.exists():
                    current = hashlib.sha256(destination.read_bytes()).hexdigest()
                    if manifest.get(name, {}).get("sha256") != current:
                        raise RuntimeError(f"Existing file is unverified or changed: {destination}")
                    if manifest[name].get("etag") != index["etag"]:
                        raise RuntimeError("Archive changed; use a new data directory.")
                    continue
                if info["size"] > 300 * 1024**2:
                    raise RuntimeError("Member exceeds the 300 MiB download limit.")
                content = range_read(session, info["offset"], info["size"], index["etag"])
                if content[:2] != b"\x1f\x8b":
                    raise RuntimeError("Expected a gzip-compressed NIfTI file.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                handle, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write(content)
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                manifest[name] = {"sha256": hashlib.sha256(content).hexdigest(),
                                  "bytes": len(content), "etag": index["etag"], "url": ARCHIVE}
                write_json(manifest_path, manifest)
            print(f"Fetched {case_id}", flush=True)
    return requested
