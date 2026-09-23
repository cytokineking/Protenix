"""Compare complete Protenix prediction artifacts for an ablation pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np


def _prediction_files(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.rglob("*sample_0.*") if path.is_file()}


def _numeric_leaves(value):
    if isinstance(value, dict):
        for subvalue in value.values():
            yield from _numeric_leaves(subvalue)
    elif isinstance(value, list):
        for subvalue in value:
            yield from _numeric_leaves(subvalue)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def _cif_coords(path: Path) -> np.ndarray:
    structure = gemmi.read_structure(str(path))
    return np.asarray([
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for model in structure for chain in model for residue in chain for atom in residue
    ], dtype=np.float64)


def compare(before_root: Path, after_root: Path) -> dict:
    before = _prediction_files(before_root)
    after = _prediction_files(after_root)
    if before.keys() != after.keys():
        raise ValueError(f"artifact names differ: {before.keys() ^ after.keys()}")
    result = {}
    for name in sorted(before):
        if name.endswith(".cif"):
            left, right = _cif_coords(before[name]), _cif_coords(after[name])
            if left.shape != right.shape:
                raise ValueError(f"coordinate shape differs: {left.shape} vs {right.shape}")
            delta = np.linalg.norm(left - right, axis=-1)
            result[name] = {
                "atoms": int(left.shape[0]),
                "coordinate_rmsd_angstrom": float(np.sqrt(np.mean(delta ** 2))),
                "max_atom_displacement_angstrom": float(delta.max()),
            }
        elif name.endswith(".json"):
            left = json.loads(before[name].read_text())
            right = json.loads(after[name].read_text())
            if left.keys() != right.keys():
                raise ValueError(f"JSON fields differ for {name}: {left.keys() ^ right.keys()}")
            fields = {}
            for key in left:
                left_values = list(_numeric_leaves(left[key]))
                right_values = list(_numeric_leaves(right[key]))
                if len(left_values) != len(right_values):
                    raise ValueError(f"numeric size differs for {name}:{key}")
                if left_values:
                    delta = np.abs(np.asarray(left_values) - np.asarray(right_values))
                    fields[key] = float(delta.max())
            result[name] = fields
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.before, args.after), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
