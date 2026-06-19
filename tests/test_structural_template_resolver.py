import tempfile
import unittest
from pathlib import Path

from protenix.data.template.structural_template import (
    StructuralTemplateError,
    features_from_template_spec_for_chain,
    parse_template_file,
    resolve_task_structural_templates,
)
from protenix.data.template.template_featurizer import (
    get_safe_entity_id_for_template_copy,
)
from protenix.data.template.template_utils import TemplateHitFeaturizer


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4} {res_name:>3} {chain_id:1}"
        f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        "  1.00 20.00           "
        f"{atom_name[0]:>2}"
    )


def _write_pdb(path: Path, chains: dict[str, list[str]]) -> None:
    aa3 = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
    }
    lines = []
    serial = 1
    chain_offset = 0.0
    for chain_id, sequence in chains.items():
        for res_idx, residue in enumerate(sequence, start=1):
            res_name = aa3[residue]
            x = chain_offset + res_idx * 3.8
            for atom_name, dx, dy, dz in [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.2, 0.1, 0.0),
                ("C", 2.4, 0.0, 0.1),
                ("O", 3.1, 0.0, 0.1),
            ]:
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        res_name,
                        chain_id,
                        res_idx,
                        x + dx,
                        dy,
                        dz,
                    )
                )
                serial += 1
        lines.append("TER")
        chain_offset += 50.0
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _write_pdb_with_residue_names(path: Path, chains: dict[str, list[str]]) -> None:
    lines = []
    serial = 1
    chain_offset = 0.0
    for chain_id, residues in chains.items():
        for res_idx, res_name in enumerate(residues, start=1):
            x = chain_offset + res_idx * 3.8
            for atom_name, dx, dy, dz in [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.2, 0.1, 0.0),
                ("C", 2.4, 0.0, 0.1),
                ("O", 3.1, 0.0, 0.1),
            ]:
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        res_name,
                        chain_id,
                        res_idx,
                        x + dx,
                        dy,
                        dz,
                    )
                )
                serial += 1
        lines.append("TER")
        chain_offset += 50.0
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


class TestStructuralTemplateResolver(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.featurizer = TemplateHitFeaturizer(
            mmcif_dir=str(self.tmp_path),
            template_cache_dir=None,
            kalign_binary_path=None,
            _zero_center_positions=True,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_direct_pdb_template_for_single_chain(self):
        pdb_path = self.tmp_path / "single.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG")})

        resolved = features_from_template_spec_for_chain(
            template_spec={"pdb": str(pdb_path)},
            query_sequence="ACDEFG",
            hit_processor=self.featurizer._hit_processor,
            template_index=0,
        )

        feature = resolved.features
        self.assertEqual(feature["template_all_atom_positions"].shape, (6, 37, 3))
        self.assertEqual(feature["template_all_atom_masks"].shape, (6, 37))
        self.assertGreater(feature["template_all_atom_masks"].sum(), 0)
        self.assertEqual(resolved.hit.aligned_cols, 6)

    def test_task_level_multichain_template_with_explicit_mapping(self):
        pdb_path = self.tmp_path / "multi.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG"), "L": list("HIKLAC")})

        resolved = resolve_task_structural_templates(
            template_specs=[
                {
                    "pdb": str(pdb_path),
                    "chain_id": ["A", "B"],
                    "template_id": ["H", "L"],
                }
            ],
            target_chains={"A": "ACDEFG", "B": "HIKLAC"},
            hit_processor=self.featurizer._hit_processor,
        )

        self.assertEqual(set(resolved), {"A", "B"})
        self.assertEqual(len(resolved["A"]), 1)
        self.assertEqual(len(resolved["B"]), 1)

    def test_ambiguous_assignment_requires_template_id(self):
        pdb_path = self.tmp_path / "ambiguous.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG"), "L": list("ACDEFG")})

        with self.assertRaisesRegex(StructuralTemplateError, "ambiguous"):
            resolve_task_structural_templates(
                template_specs=[{"pdb": str(pdb_path), "chain_id": ["A"]}],
                target_chains={"A": "ACDEFG"},
                hit_processor=self.featurizer._hit_processor,
            )

    def test_duplicate_template_indices_are_rejected(self):
        pdb_path = self.tmp_path / "duplicate_indices.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG")})

        with self.assertRaisesRegex(StructuralTemplateError, "Duplicate template index"):
            features_from_template_spec_for_chain(
                template_spec={
                    "pdb": str(pdb_path),
                    "queryIndices": [0, 1],
                    "templateIndices": [0, 0],
                },
                query_sequence="ACDEFG",
                hit_processor=self.featurizer._hit_processor,
                template_index=0,
            )

    def test_pdb_parser_ignores_non_protein_polymer_chains(self):
        pdb_path = self.tmp_path / "protein_and_dna.pdb"
        _write_pdb_with_residue_names(
            pdb_path,
            {"A": ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY"], "B": ["DA", "DT"]},
        )

        mmcif_object = parse_template_file(str(pdb_path))

        self.assertEqual(set(mmcif_object.chain_to_seqres), {"A"})

    def test_task_level_templates_disable_entity_template_copy(self):
        bioassembly = {
            0: {
                "entity_id": 0,
                "sequence": "ACDEFG",
                "copy_templates": False,
            },
            1: {
                "entity_id": 0,
                "sequence": "ACDEFG",
                "copy_templates": True,
            },
        }

        self.assertEqual(get_safe_entity_id_for_template_copy(bioassembly), [])


if __name__ == "__main__":
    unittest.main()
