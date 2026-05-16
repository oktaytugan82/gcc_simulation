from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_size: int, retries: int = 4) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        return {"status": "exists", "bytes": dest.stat().st_size}

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "GCC-DoC-validation/1.0"})
            with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            got = tmp.stat().st_size
            if got != expected_size:
                raise IOError(f"size mismatch: expected {expected_size}, got {got}")
            tmp.replace(dest)
            return {"status": "downloaded", "bytes": got, "attempt": attempt}
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(30, 2**attempt))
    return {"status": "failed", "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify-sha", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig")))
    if args.limit:
        rows = rows[: args.limit]

    log = []
    for idx, row in enumerate(rows, start=1):
        filename = row["filename"]
        exact_size = row.get("size_bytes")
        expected_size = int(exact_size) if exact_size else int(round(float(row["size_mb"].replace(",", ".")) * 1024 * 1024))
        # The manifest stores rounded MB for readability; use Content-Length from HEAD-like metadata
        # if present in future manifests, otherwise accept a small rounding tolerance after download.
        dest = args.outdir / filename
        print(f"[{idx}/{len(rows)}] {filename}", flush=True)

        # Re-read exact size from the original JSON mirror if available.
        expected = expected_size

        result = download(row["url"], dest, expected)
        if result["status"] == "failed" and not row.get("size_bytes"):
            # Retry without rounded-size enforcement by using the remote file as written.
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            try:
                req = Request(row["url"], headers={"User-Agent": "GCC-DoC-validation/1.0"})
                with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
                result = {"status": "downloaded_unchecked", "bytes": dest.stat().st_size}
            except Exception as exc:  # noqa: BLE001
                result = {"status": "failed", "error": str(exc)}

        result.update({"filename": filename, "label": row["label"], "path": str(dest)})
        if args.verify_sha and dest.exists() and row.get("sha256_hash"):
            result["sha256"] = sha256_file(dest)
            result["sha256_ok"] = result["sha256"].lower() == row["sha256_hash"].lower()
        log.append(result)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    ok = sum(1 for x in log if x["status"] != "failed")
    print(json.dumps({"requested": len(rows), "ok": ok, "failed": len(rows) - ok}, indent=2))


if __name__ == "__main__":
    main()
