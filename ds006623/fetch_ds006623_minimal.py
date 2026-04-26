"""Targeted downloader for the OpenNeuro ds006623 fMRI validation subset.

The full dataset is large. This script uses the public OpenNeuro S3 mirror and
downloads only the files needed for the GCC validation plan:

* core dataset metadata
* LOR/ROR and participant tables
* propofol infusion and squeeze-force behavioral files
* XCP-D ROI mean time series, motion, QC, and coverage files for a selected
  denoising pipeline and atlas

No AWS CLI, boto3, DataLad, or OpenNeuro account is required.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


S3_LIST_URL = "https://s3.amazonaws.com/openneuro.org/"
S3_OBJECT_URL = "https://s3.amazonaws.com/openneuro.org/{key}"
DATASET = "ds006623"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str
    etag: str


def request_url(url: str, *, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gcc-ds006623-validator/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def list_prefix(prefix: str) -> list[S3Object]:
    """List all objects under an S3 prefix."""
    objects: list[S3Object] = []
    token: str | None = None
    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if token:
            params["continuation-token"] = token
        url = S3_LIST_URL + "?" + urllib.parse.urlencode(params)
        root = ET.fromstring(request_url(url))
        for item in root.findall("s3:Contents", XML_NS):
            key = item.findtext("s3:Key", default="", namespaces=XML_NS)
            size = int(item.findtext("s3:Size", default="0", namespaces=XML_NS))
            last_modified = item.findtext("s3:LastModified", default="", namespaces=XML_NS)
            etag = item.findtext("s3:ETag", default="", namespaces=XML_NS).strip('"')
            if key:
                objects.append(S3Object(key=key, size=size, last_modified=last_modified, etag=etag))
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=XML_NS).lower() == "true"
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=XML_NS)
        if not truncated:
            break
    return objects


def build_candidate_manifest(pipeline: str, atlas: str) -> list[S3Object]:
    prefixes = [
        f"{DATASET}/",
        f"{DATASET}/derivatives/Propofol_Infusion/",
        f"{DATASET}/derivatives/Squeeze_Force/",
        f"{DATASET}/derivatives/Stimulus_Timing/",
        f"{DATASET}/derivatives/{pipeline}/",
    ]

    seen: dict[str, S3Object] = {}
    for prefix in prefixes:
        for obj in list_prefix(prefix):
            seen[obj.key] = obj

    include_exact = {
        f"{DATASET}/CHANGES",
        f"{DATASET}/README.md",
        f"{DATASET}/dataset_description.json",
        f"{DATASET}/derivatives/LOR_ROR_Timing.csv",
        f"{DATASET}/derivatives/LOR_ROR_Timing.xlsx",
        f"{DATASET}/derivatives/Participant_Info.csv",
        f"{DATASET}/derivatives/Participant_Info.xlsx",
    }
    include_patterns = [
        f"{DATASET}/code/*.md",
        f"{DATASET}/derivatives/Propofol_Infusion/**/*.1D",
        f"{DATASET}/derivatives/Propofol_Infusion/**/*.csv",
        f"{DATASET}/derivatives/Propofol_Infusion/**/*.tsv",
        f"{DATASET}/derivatives/Squeeze_Force/**/*.1D",
        f"{DATASET}/derivatives/Squeeze_Force/**/*.csv",
        f"{DATASET}/derivatives/Squeeze_Force/**/*.tsv",
        f"{DATASET}/derivatives/Stimulus_Timing/**/*.1D",
        f"{DATASET}/derivatives/Stimulus_Timing/**/*.csv",
        f"{DATASET}/derivatives/Stimulus_Timing/**/*.tsv",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_motion.tsv",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_motion.json",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_desc-linc_qc.tsv",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_seg-{atlas}_stat-mean_timeseries.tsv",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_seg-{atlas}_stat-mean_timeseries.json",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_seg-{atlas}_stat-coverage_bold.tsv",
        f"{DATASET}/derivatives/{pipeline}/sub-*/func/*_seg-{atlas}_stat-coverage_bold.json",
    ]

    selected: list[S3Object] = []
    for obj in seen.values():
        if obj.key in include_exact or any(fnmatch.fnmatch(obj.key, pat) for pat in include_patterns):
            selected.append(obj)
    return sorted(selected, key=lambda item: item.key)


def write_manifest(objects: list[S3Object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "size", "last_modified", "etag"])
        writer.writeheader()
        for obj in objects:
            writer.writerow(
                {
                    "key": obj.key,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                }
            )


def download_object(obj: S3Object, target_root: Path, *, overwrite: bool) -> bool:
    relative_key = obj.key.removeprefix(f"{DATASET}/")
    target = target_root / relative_key
    if target.exists() and target.stat().st_size == obj.size and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    url = S3_OBJECT_URL.format(key=urllib.parse.quote(obj.key, safe="/"))
    tmp = target.with_suffix(target.suffix + ".part")
    data = request_url(url)
    tmp.write_bytes(data)
    if tmp.stat().st_size != obj.size:
        raise RuntimeError(f"size mismatch for {obj.key}: expected {obj.size}, got {tmp.stat().st_size}")
    os.replace(tmp, target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-root",
        default=str(Path("data") / "ds006623-minimal"),
        help="Local dataset subset root.",
    )
    parser.add_argument(
        "--pipeline",
        default="xcp_d_without_GSR_bandpass_output",
        help="XCP-D derivatives pipeline to use as primary analysis input.",
    )
    parser.add_argument(
        "--atlas",
        default="4S156Parcels",
        help="Atlas segment identifier used in XCP-D filenames.",
    )
    parser.add_argument("--manifest-only", action="store_true", help="Only write the manifest; do not download files.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even when size matches.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for number of files to download.")
    parser.add_argument("--quiet", action="store_true", help="Print compact progress instead of one line per file.")
    args = parser.parse_args()

    target_root = Path(args.target_root)
    manifest_path = target_root / f"ds006623_minimal_manifest_{args.pipeline}_{args.atlas}.csv"
    objects = build_candidate_manifest(args.pipeline, args.atlas)
    write_manifest(objects, manifest_path)
    write_manifest(objects, target_root / "ds006623_minimal_manifest.csv")

    total_size = sum(obj.size for obj in objects)
    print(f"selected_files={len(objects)}")
    print(f"selected_size_mb={total_size / 1024 / 1024:.1f}")
    print(f"manifest={manifest_path}")

    if args.manifest_only:
        return 0

    to_download = objects[: args.limit] if args.limit else objects
    downloaded = 0
    skipped = 0
    for index, obj in enumerate(to_download, start=1):
        changed = download_object(obj, target_root, overwrite=args.overwrite)
        if changed:
            downloaded += 1
            action = "downloaded"
        else:
            skipped += 1
            action = "skipped"
        if args.quiet:
            if index == 1 or index == len(to_download) or index % 100 == 0:
                print(f"progress={index}/{len(to_download)} downloaded={downloaded} skipped={skipped}")
        else:
            print(f"[{index}/{len(to_download)}] {action}: {obj.key} ({obj.size} bytes)")

    print(f"downloaded={downloaded}")
    print(f"skipped={skipped}")
    print(f"target_root={target_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
