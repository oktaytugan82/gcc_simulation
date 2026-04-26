"""Prepare Budapest Reference Connectome files for GCC simulations.

The script converts the public HCP-derived Budapest Reference Connectome
CSV/GraphML downloads into analysis-ready NumPy matrices and lightweight
metadata tables. It is intentionally deterministic and does not download data.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BUDAPEST_CONNECTOME_DIR", ROOT / "data" / "budapest_connectome"))
GRAPHML = DATA_DIR / "brc_v3_default_20k_fibercount_median.graphml"


VARIANTS = [
    {
        "id": "brc_v3_20k_fibercount_conf50_default",
        "file": "brc_v3_default_20k_fibercount_median.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 418,
        "min_confidence_percent": 50,
        "min_occurrences": 209,
        "weight_function": "fiber_count",
        "combine_mode": "median",
        "total_fiber_number": "20k",
    },
    {
        "id": "brc_v3_20k_fibercount_conf25",
        "file": "brc_v3_20k_fibercount_median_conf25.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 418,
        "min_confidence_percent": 25,
        "min_occurrences": 105,
        "weight_function": "fiber_count",
        "combine_mode": "median",
        "total_fiber_number": "20k",
    },
    {
        "id": "brc_v3_20k_fibercount_conf10",
        "file": "brc_v3_20k_fibercount_median_conf10.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 418,
        "min_confidence_percent": 10,
        "min_occurrences": 42,
        "weight_function": "fiber_count",
        "combine_mode": "median",
        "total_fiber_number": "20k",
    },
    {
        "id": "brc_v3_200k_fibercount_conf50",
        "file": "brc_v3_200k_fibercount_median_conf50.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 477,
        "min_confidence_percent": 50,
        "min_occurrences": 239,
        "weight_function": "fiber_count",
        "combine_mode": "median",
        "total_fiber_number": "200k",
    },
    {
        "id": "brc_v3_1m_fibercount_conf50",
        "file": "brc_v3_1m_fibercount_median_conf50.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 476,
        "min_confidence_percent": 50,
        "min_occurrences": 238,
        "weight_function": "fiber_count",
        "combine_mode": "median",
        "total_fiber_number": "1m",
    },
    {
        "id": "brc_v3_20k_electrical_conf50",
        "file": "brc_v3_20k_electrical_median_conf50.csv",
        "version": "3.0",
        "population": "all",
        "subjects_total": 418,
        "min_confidence_percent": 50,
        "min_occurrences": 209,
        "weight_function": "electrical_connectivity_n_over_length",
        "combine_mode": "median",
        "total_fiber_number": "20k",
    },
]


def parse_nodes(graphml_path: Path) -> list[dict[str, str]]:
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(graphml_path).getroot()

    key_to_attr: dict[str, str] = {}
    for key in root.findall("g:key", ns):
        key_id = key.attrib["id"]
        key_to_attr[key_id] = key.attrib.get("attr.name", key_id)

    rows: list[dict[str, str]] = []
    for node in root.findall(".//g:node", ns):
        row = {"node_id": node.attrib["id"]}
        for data in node.findall("g:data", ns):
            row[key_to_attr[data.attrib["key"]]] = data.text or ""
        rows.append(row)

    rows.sort(key=lambda item: int(item["node_id"]))
    return rows


def write_node_metadata(nodes: list[dict[str, str]]) -> Path:
    out_path = DATA_DIR / "budapest_connectome_node_metadata.csv"
    fieldnames = ["node_id", "dn_name", "dn_fsname", "dn_region", "dn_hemisphere"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in nodes:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return out_path


def variant_url(variant: dict[str, object], fmt: str = "csv") -> str:
    weight_function_code = {
        "electrical_connectivity_n_over_length": 0,
        "fiber_count": 1,
        "fiber_length": 2,
        "fractional_anisotropy": 3,
    }[str(variant["weight_function"])]
    fiber_code = {"20k": 0, "200k": 1, "1m": 2}[str(variant["total_fiber_number"])]
    return (
        "https://pitgroup.org/apps/connectome/getgraph.php"
        f"?format={fmt}"
        "&version=2"
        "&population=0"
        f"&minOccurrences={variant['min_occurrences']}"
        "&minStrength=0"
        f"&combineMode={variant['combine_mode']}"
        f"&weightFunction={weight_function_code}"
        f"&totalFiberNumber={fiber_code}"
    )


def read_edge_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames or []
        weight_columns = [name for name in fieldnames if name.startswith("edge weight")]
        if len(weight_columns) != 1:
            raise ValueError(f"Expected exactly one edge-weight column in {path}, found {weight_columns}")
        return list(reader), weight_columns[0]


def connected_components(binary: np.ndarray) -> list[int]:
    n = binary.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes: list[int] = []
    neighbors = [np.flatnonzero(binary[i]) for i in range(n)]
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nbr in neighbors[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(int(nbr))
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def matrix_stats(
    matrix: np.ndarray,
    confidence: np.ndarray,
    input_edge_rows: int,
    self_loop_rows: int,
    duplicate_offdiag_rows: int,
) -> dict[str, object]:
    n = matrix.shape[0]
    triu = np.triu_indices(n, k=1)
    weights = matrix[triu]
    confidences = confidence[triu]
    nonzero = weights > 0
    binary = (matrix > 0).astype(np.uint8)
    components = connected_components(binary)
    degrees = binary.sum(axis=0)
    strengths = matrix.sum(axis=0)
    possible_edges = n * (n - 1) / 2

    return {
        "nodes": int(n),
        "input_edge_rows": int(input_edge_rows),
        "self_loop_rows_excluded": int(self_loop_rows),
        "duplicate_offdiag_rows_collapsed": int(duplicate_offdiag_rows),
        "edges": int(nonzero.sum()),
        "density": float(nonzero.sum() / possible_edges),
        "nonisolated_nodes": int((degrees > 0).sum()),
        "isolated_nodes": int((degrees == 0).sum()),
        "largest_component_nodes": int(components[0] if components else 0),
        "component_count": int(len(components)),
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
        "strength_mean": float(strengths.mean()),
        "strength_median": float(np.median(strengths)),
        "strength_max": float(strengths.max()),
        "edge_weight_mean": float(weights[nonzero].mean()) if nonzero.any() else math.nan,
        "edge_weight_median": float(np.median(weights[nonzero])) if nonzero.any() else math.nan,
        "edge_weight_max": float(weights[nonzero].max()) if nonzero.any() else math.nan,
        "edge_confidence_mean": float(confidences[nonzero].mean()) if nonzero.any() else math.nan,
        "edge_confidence_median": float(np.median(confidences[nonzero])) if nonzero.any() else math.nan,
    }


def safe_npz_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nodes = parse_nodes(GRAPHML)
    if len(nodes) != 1015:
        raise ValueError(f"Expected 1015 nodes in GraphML, found {len(nodes)}")
    node_ids = [int(row["node_id"]) for row in nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_metadata_path = write_node_metadata(nodes)

    arrays: dict[str, np.ndarray] = {}
    stats_rows: list[dict[str, object]] = []
    manifest_variants: list[dict[str, object]] = []

    for variant in VARIANTS:
        csv_path = DATA_DIR / str(variant["file"])
        edges, weight_col = read_edge_csv(csv_path)
        matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        confidence = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        seen_pairs: set[tuple[int, int]] = set()
        self_loop_rows = 0
        duplicate_offdiag_rows = 0

        for edge in edges:
            node1 = int(edge["id node1"])
            node2 = int(edge["id node2"])
            if node1 == node2:
                self_loop_rows += 1
                continue
            pair = tuple(sorted((node1, node2)))
            if pair in seen_pairs:
                duplicate_offdiag_rows += 1
            seen_pairs.add(pair)

            i = id_to_index[node1]
            j = id_to_index[node2]
            weight = float(edge[weight_col])
            conf = float(edge["edge confidence"])
            matrix[i, j] = matrix[j, i] = weight
            confidence[i, j] = confidence[j, i] = conf

        key = safe_npz_key(str(variant["id"]))
        arrays[f"{key}_weight"] = matrix
        arrays[f"{key}_confidence"] = confidence
        stats = matrix_stats(
            matrix,
            confidence,
            input_edge_rows=len(edges),
            self_loop_rows=self_loop_rows,
            duplicate_offdiag_rows=duplicate_offdiag_rows,
        )
        stats_rows.append({"variant_id": variant["id"], **stats})
        manifest_variants.append(
            {
                **variant,
                "csv_path": str(csv_path),
                "source_url": variant_url(variant, "csv"),
                "edge_weight_column": weight_col,
                "stats": stats,
            }
        )

    npz_path = DATA_DIR / "budapest_connectome_matrices.npz"
    np.savez_compressed(
        npz_path,
        node_ids=np.asarray(node_ids, dtype=np.int32),
        **arrays,
    )

    stats_path = DATA_DIR / "budapest_connectome_variant_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(stats_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    manifest = {
        "dataset": "Budapest Reference Connectome v3.0",
        "description": "HCP-derived consensus structural connectomes from diffusion MRI.",
        "source_page": "https://pitgroup.org/connectome/",
        "source_article_v2": "https://doi.org/10.1016/j.neulet.2015.03.071",
        "source_article_v3": "https://doi.org/10.1007/s11571-016-9407-z",
        "graphml_path": str(GRAPHML),
        "node_metadata_path": str(node_metadata_path),
        "matrix_npz_path": str(npz_path),
        "variant_stats_path": str(stats_path),
        "variants": manifest_variants,
    }
    manifest_path = DATA_DIR / "budapest_connectome_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared {len(VARIANTS)} variants")
    print(f"Nodes: {len(nodes)}")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
