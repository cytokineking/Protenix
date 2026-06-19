# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for user-supplied structural templates."""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from Bio import Align, PDB
from Bio.Data import PDBData

from protenix.data.template.template_parser import (
    MmcifObject,
    ResidueAtPosition,
    ResiduePosition,
    TemplateHit,
    TemplateParser,
)


class StructuralTemplateError(ValueError):
    """Raised when a user-supplied structural template cannot be resolved."""


@dataclass(frozen=True)
class StructuralTemplateFeatures:
    """Resolved template features for one target chain."""

    chain_id: str
    features: Mapping[str, Any]
    hit: TemplateHit
    warnings: tuple[str, ...] = ()


def _as_list(value: Any, field_name: str) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        if not all(isinstance(x, (str, int)) for x in value):
            raise StructuralTemplateError(f"`{field_name}` must contain strings.")
        return [str(x) for x in value]
    if isinstance(value, (str, int)):
        return [str(value)]
    raise StructuralTemplateError(f"`{field_name}` must be a string or list of strings.")


def _resolve_path(path: str, base_path: Optional[str] = None) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        raise StructuralTemplateError(f"Template path does not exist: {path}")

    if base_path:
        base_dir = Path(base_path).expanduser()
        if base_dir.is_file():
            base_dir = base_dir.parent
        candidate = base_dir / path
        if candidate.exists():
            return str(candidate)

    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate)

    raise StructuralTemplateError(f"Template path does not exist: {path}")


def resolve_maybe_relative_path(path: str, base_path: Optional[str] = None) -> str:
    """Resolve a template path using the input JSON directory, then current directory."""

    return _resolve_path(path, base_path)


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _template_name_from_path(path: str) -> str:
    return Path(path).stem or "template"


def _parse_mmcif_string(
    *, file_id: str, mmcif_string: str, allow_simple_fallback: bool = True
) -> MmcifObject:
    result = TemplateParser.parse(file_id=file_id, mmcif_string=mmcif_string)
    if result.mmcif_object is not None:
        return result.mmcif_object

    if allow_simple_fallback:
        simple = TemplateParser.parse_simple_cif(
            file_id=file_id, mmcif_string=mmcif_string
        )
        if simple.mmcif_object is not None:
            return simple.mmcif_object
        errors = {**result.errors, **simple.errors}
    else:
        errors = result.errors

    raise StructuralTemplateError(f"Failed to parse mmCIF template {file_id}: {errors}")


def _parse_pdb_file(path: str, file_id: str) -> MmcifObject:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(file_id, path)
    first_model = next(structure.get_models())

    chain_to_seqres = {}
    seqres_to_structure = collections.defaultdict(dict)

    for chain in first_model.get_chains():
        sequence = []
        for residue in chain.get_residues():
            hetflag, resseq, insertion_code = residue.get_id()
            resname = residue.get_resname()
            is_standard_protein_residue = (
                hetflag == " " and resname in PDBData.protein_letters_3to1
            )
            is_supported_modified_residue = resname == "MSE"
            if not is_standard_protein_residue and not is_supported_modified_residue:
                continue

            one_letter = PDBData.protein_letters_3to1.get(resname, "X")
            if resname == "MSE":
                one_letter = "M"

            seq_idx = len(sequence)
            sequence.append(one_letter if len(one_letter) == 1 else "X")
            seqres_to_structure[chain.id][seq_idx] = ResidueAtPosition(
                position=ResiduePosition(
                    chain_id=chain.id,
                    residue_number=int(resseq),
                    insertion_code=insertion_code if insertion_code.strip() else " ",
                ),
                name=resname,
                is_missing=False,
                hetflag=hetflag,
            )

        if sequence:
            chain_to_seqres[chain.id] = "".join(sequence)

    if not chain_to_seqres:
        raise StructuralTemplateError(f"No protein chains found in PDB template: {path}")

    return MmcifObject(
        file_id=file_id,
        header={},
        structure=first_model,
        chain_to_seqres=chain_to_seqres,
        seqres_to_structure=seqres_to_structure,
        raw_string={},
    )


def parse_template_file(path: str) -> MmcifObject:
    """Parse a CIF/mmCIF/PDB template into Protenix's template parser object."""

    file_id = _template_name_from_path(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return _parse_mmcif_string(file_id=file_id, mmcif_string=_read_text(path))
    if suffix == ".pdb":
        return _parse_pdb_file(path=path, file_id=file_id)
    raise StructuralTemplateError(f"Unsupported structural template format: {path}")


def _template_spec_to_mmcif_object(
    template_spec: Mapping[str, Any],
    base_path: Optional[str],
    default_name: str,
) -> tuple[MmcifObject, str]:
    inline_mmcif = template_spec.get("mmcif")
    path_fields = [
        ("cif", "cif"),
        ("mmcifPath", "cif"),
        ("cifPath", "cif"),
        ("pdb", "pdb"),
        ("pdbPath", "pdb"),
    ]
    present_paths = [(field, kind) for field, kind in path_fields if template_spec.get(field)]

    if inline_mmcif and present_paths:
        raise StructuralTemplateError(
            "Set either inline `mmcif` or a template path, not both."
        )
    if len(present_paths) > 1:
        fields = ", ".join(field for field, _ in present_paths)
        raise StructuralTemplateError(f"Only one template path field may be set: {fields}")

    if inline_mmcif:
        return (
            _parse_mmcif_string(file_id=default_name, mmcif_string=inline_mmcif),
            default_name,
        )

    if not present_paths:
        raise StructuralTemplateError(
            "Template entry must set one of `mmcif`, `cif`, `pdb`, "
            "`mmcifPath`, `cifPath`, or `pdbPath`."
        )

    field, _kind = present_paths[0]
    path = _resolve_path(str(template_spec[field]), base_path)
    return parse_template_file(path), _template_name_from_path(path)


def _alignment_score(query: str, template: str) -> float:
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    return float(aligner.align(query, template)[0].score)


def _align_query_to_template_indices(
    query_sequence: str, template_sequence: str, min_similarity: float = 0.0
) -> dict[int, int]:
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    alignment = aligner.align(query_sequence, template_sequence)[0]

    mapping: dict[int, int] = {}
    same = 0
    count = 0
    for q_block, t_block in zip(alignment.aligned[0], alignment.aligned[1]):
        q_start, q_end = map(int, q_block)
        t_start, t_end = map(int, t_block)
        block_len = min(q_end - q_start, t_end - t_start)
        for offset in range(block_len):
            q_idx = q_start + offset
            t_idx = t_start + offset
            mapping[q_idx] = t_idx
            count += 1
            if query_sequence[q_idx] == template_sequence[t_idx]:
                same += 1

    if not mapping:
        raise StructuralTemplateError("Template sequence does not align to query.")
    if count and same / count < min_similarity:
        raise StructuralTemplateError("Insufficient similarity to query.")

    return mapping


def _normalize_explicit_mapping(
    *,
    query_indices: Sequence[Any],
    template_indices: Sequence[Any],
    query_sequence: str,
    template_sequence: str,
) -> dict[int, int]:
    if len(query_indices) != len(template_indices):
        raise StructuralTemplateError(
            "`queryIndices` and `templateIndices` must have the same length."
        )

    mapping = {}
    seen_template_indices = set()
    for raw_q, raw_t in zip(query_indices, template_indices):
        q_idx = int(raw_q)
        t_idx = int(raw_t)
        if q_idx in mapping:
            raise StructuralTemplateError(f"Duplicate query index in template: {q_idx}")
        if t_idx in seen_template_indices:
            raise StructuralTemplateError(f"Duplicate template index in template: {t_idx}")
        if q_idx < 0 or q_idx >= len(query_sequence):
            raise StructuralTemplateError(
                f"query index {q_idx} out of range for query length {len(query_sequence)}"
            )
        if t_idx < 0 or t_idx >= len(template_sequence):
            raise StructuralTemplateError(
                f"template index {t_idx} out of range for template length {len(template_sequence)}"
            )
        mapping[q_idx] = t_idx
        seen_template_indices.add(t_idx)

    if not mapping:
        raise StructuralTemplateError("Explicit template mapping is empty.")
    return mapping


def _hit_from_mapping(
    *,
    index: int,
    name: str,
    query_sequence: str,
    template_sequence: str,
    mapping: Mapping[int, int],
) -> TemplateHit:
    indices_hit = [-1] * len(query_sequence)
    for q_idx, t_idx in mapping.items():
        indices_hit[q_idx] = t_idx
    return TemplateHit(
        index=index,
        name=name,
        aligned_cols=len(mapping),
        sum_probs=1.0,
        query=query_sequence,
        hit_sequence=template_sequence,
        indices_query=list(range(len(query_sequence))),
        indices_hit=indices_hit,
    )


def _extract_chain_features(
    *,
    hit_processor: Any,
    mmcif_object: MmcifObject,
    template_name: str,
    query_sequence: str,
    template_chain_id: str,
    mapping: Mapping[int, int],
    feature_index: int,
) -> StructuralTemplateFeatures:
    template_sequence = mmcif_object.chain_to_seqres[template_chain_id]
    features, warning = hit_processor._extract_template_features(  # noqa: SLF001
        mmcif_object,
        template_name,
        mapping,
        template_sequence,
        query_sequence,
        template_chain_id,
        hit_processor._zero_center_positions,  # noqa: SLF001
    )

    release_date = mmcif_object.header.get("release_date") or "9999-12-31"
    warnings = []
    if "release_date" not in mmcif_object.header:
        warnings.append(
            f"Template {template_name}_{template_chain_id} has no release date; "
            "using 9999-12-31."
        )
    if warning:
        warnings.append(warning)

    features["template_sum_probs"] = [1.0]
    features["template_release_date"] = np.array(release_date.encode(), dtype=object)
    hit = _hit_from_mapping(
        index=feature_index,
        name=f"{template_name}_{template_chain_id}",
        query_sequence=query_sequence,
        template_sequence=template_sequence,
        mapping=mapping,
    )
    return StructuralTemplateFeatures(
        chain_id="",
        features=features,
        hit=hit,
        warnings=tuple(warnings),
    )


def features_from_template_spec_for_chain(
    *,
    template_spec: Mapping[str, Any],
    query_sequence: str,
    hit_processor: Any,
    template_index: int,
    base_path: Optional[str] = None,
    default_name: Optional[str] = None,
) -> StructuralTemplateFeatures:
    """Create template features for one target chain from an inline/file spec."""

    template_name = default_name or f"template_{template_index}"
    mmcif_object, parsed_name = _template_spec_to_mmcif_object(
        template_spec=template_spec,
        base_path=base_path,
        default_name=template_name,
    )
    template_name = parsed_name or template_name

    template_chain_ids = _as_list(
        template_spec.get("template_id", template_spec.get("templateChainId")),
        "template_id",
    )
    if template_chain_ids is None:
        if len(mmcif_object.chain_to_seqres) == 1:
            template_chain_id = next(iter(mmcif_object.chain_to_seqres))
        else:
            scores = {
                chain_id: _alignment_score(query_sequence, sequence)
                for chain_id, sequence in mmcif_object.chain_to_seqres.items()
            }
            best_score = max(scores.values())
            best_chains = [
                chain_id
                for chain_id, score in scores.items()
                if np.isclose(score, best_score)
            ]
            if len(best_chains) != 1:
                raise StructuralTemplateError(
                    "Template chain assignment is ambiguous; provide `template_id`."
                )
            template_chain_id = best_chains[0]
    elif len(template_chain_ids) == 1:
        template_chain_id = template_chain_ids[0]
    else:
        raise StructuralTemplateError(
            "Per-chain template JSON entries may specify only one `template_id`."
        )

    if template_chain_id not in mmcif_object.chain_to_seqres:
        raise StructuralTemplateError(
            f"Template chain {template_chain_id} not found in template {template_name}."
        )

    template_sequence = mmcif_object.chain_to_seqres[template_chain_id]
    q_indices = template_spec.get("queryIndices")
    t_indices = template_spec.get("templateIndices")
    if q_indices is not None or t_indices is not None:
        if q_indices is None or t_indices is None:
            raise StructuralTemplateError(
                "`queryIndices` and `templateIndices` must be provided together."
            )
        mapping = _normalize_explicit_mapping(
            query_indices=q_indices,
            template_indices=t_indices,
            query_sequence=query_sequence,
            template_sequence=template_sequence,
        )
    else:
        mapping = _align_query_to_template_indices(query_sequence, template_sequence)

    return _extract_chain_features(
        hit_processor=hit_processor,
        mmcif_object=mmcif_object,
        template_name=template_name,
        query_sequence=query_sequence,
        template_chain_id=template_chain_id,
        mapping=mapping,
        feature_index=template_index,
    )


def _infer_chain_pairs(
    *,
    target_chain_ids: Sequence[str],
    target_sequences: Mapping[str, str],
    template_chain_ids: Sequence[str],
    template_sequences: Mapping[str, str],
) -> list[tuple[str, str]]:
    if not target_chain_ids:
        return []
    if len(template_chain_ids) < len(target_chain_ids):
        raise StructuralTemplateError(
            "Not enough template protein chains to map all requested target chains."
        )

    score_matrix = np.array(
        [
            [
                _alignment_score(
                    target_sequences[target_id], template_sequences[tmpl_id]
                )
                for tmpl_id in template_chain_ids
            ]
            for target_id in target_chain_ids
        ],
        dtype=np.float64,
    )

    best_score = None
    best_assignment = None
    template_indices = range(len(template_chain_ids))
    for col_ind in itertools.permutations(template_indices, len(target_chain_ids)):
        score = sum(
            score_matrix[row_idx, col_idx]
            for row_idx, col_idx in enumerate(col_ind)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_assignment = col_ind
        elif np.isclose(score, best_score):
            best_assignment = None

    if best_assignment is None:
        raise StructuralTemplateError(
            "Template chain assignment is ambiguous; provide `chain_id` and `template_id`."
        )

    return [
        (target_chain_ids[row_idx], template_chain_ids[col_idx])
        for row_idx, col_idx in enumerate(best_assignment)
    ]


def resolve_task_structural_templates(
    *,
    template_specs: Sequence[Mapping[str, Any]],
    target_chains: Mapping[str, str],
    hit_processor: Any,
    base_path: Optional[str] = None,
) -> dict[str, list[Mapping[str, Any]]]:
    """Resolve task-level structural templates into per-target-chain features."""

    resolved: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    if not template_specs:
        return resolved

    for template_index, template_spec in enumerate(template_specs):
        if not isinstance(template_spec, Mapping):
            raise StructuralTemplateError("Each template entry must be an object.")

        mmcif_object, template_name = _template_spec_to_mmcif_object(
            template_spec=template_spec,
            base_path=base_path,
            default_name=f"template_{template_index}",
        )

        requested_targets = _as_list(template_spec.get("chain_id"), "chain_id")
        requested_templates = _as_list(template_spec.get("template_id"), "template_id")

        target_chain_ids = requested_targets or list(target_chains)
        for chain_id in target_chain_ids:
            if chain_id not in target_chains:
                raise StructuralTemplateError(
                    f"Target chain {chain_id} is not a protein chain in this sample."
                )

        template_chain_ids = requested_templates or list(mmcif_object.chain_to_seqres)
        for chain_id in template_chain_ids:
            if chain_id not in mmcif_object.chain_to_seqres:
                raise StructuralTemplateError(
                    f"Template chain {chain_id} not found in template {template_name}."
                )

        if requested_targets is not None and requested_templates is not None:
            if len(requested_targets) != len(requested_templates):
                raise StructuralTemplateError(
                    "`chain_id` and `template_id` must have the same length."
                )
            chain_pairs = list(zip(requested_targets, requested_templates))
        else:
            chain_pairs = _infer_chain_pairs(
                target_chain_ids=target_chain_ids,
                target_sequences=target_chains,
                template_chain_ids=template_chain_ids,
                template_sequences=mmcif_object.chain_to_seqres,
            )

        q_indices = template_spec.get("queryIndices")
        t_indices = template_spec.get("templateIndices")
        if (q_indices is not None or t_indices is not None) and len(chain_pairs) != 1:
            raise StructuralTemplateError(
                "Explicit residue mappings are supported only for one target/template "
                "chain pair per template entry."
            )

        for target_chain_id, template_chain_id in chain_pairs:
            query_sequence = target_chains[target_chain_id]
            template_sequence = mmcif_object.chain_to_seqres[template_chain_id]
            if q_indices is not None or t_indices is not None:
                if q_indices is None or t_indices is None:
                    raise StructuralTemplateError(
                        "`queryIndices` and `templateIndices` must be provided together."
                    )
                mapping = _normalize_explicit_mapping(
                    query_indices=q_indices,
                    template_indices=t_indices,
                    query_sequence=query_sequence,
                    template_sequence=template_sequence,
                )
            else:
                mapping = _align_query_to_template_indices(
                    query_sequence, template_sequence
                )

            features = _extract_chain_features(
                hit_processor=hit_processor,
                mmcif_object=mmcif_object,
                template_name=template_name,
                query_sequence=query_sequence,
                template_chain_id=template_chain_id,
                mapping=mapping,
                feature_index=template_index,
            )
            resolved[target_chain_id].append(features.features)

    return dict(resolved)


def is_structural_template_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".cif", ".mmcif", ".pdb"}
