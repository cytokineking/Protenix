# Protenix Fork: Confidence Scoring and Structural Templates

This repository is a fork of
[bytedance/Protenix](https://github.com/bytedance/Protenix). It keeps upstream
Protenix intact where possible and focuses on two practical workflows:

1. Score existing structures with the Protenix confidence head, without running
   diffusion.
2. Provide structural templates directly as CIF/mmCIF/PDB files, including
   multichain templates.

For upstream model details, benchmarks, training instructions, general
inference setup, and citation guidance, use the official Protenix repository and
documentation.

## What This Fork Adds

### Score Existing Structures

The fork exposes a score-only mode for the Protenix confidence head. Instead of
sampling coordinates with diffusion, the model accepts supplied coordinates and
computes Protenix confidence outputs for those structures.

This is intended to power
[ProtenixScore](https://github.com/cytokineking/ProtenixScore) workflows.

```bash
# Score a single structure.
protenix score --input examples/7pzb.cif --output ./score_out

# Score a directory of PDB/CIF files recursively.
protenix score --input ./structures --output ./score_out --recursive
```

The `protenix score` command requires the optional
[`protenixscore`](https://github.com/cytokineking/ProtenixScore) package. At the
model level, this uses `score_only=True` and supplies coordinates through
`x_pred_coords`, so diffusion sampling is skipped.

### Direct Structural Template Support

Upstream Protenix supports template search and precomputed template sidecars.
This fork adds a simpler structural-template interface inspired by Boltz:

```json
{
  "name": "template_example",
  "sequences": [
    {
      "proteinChain": {
        "id": ["A"],
        "sequence": "PREACHINGS",
        "count": 1
      }
    },
    {
      "proteinChain": {
        "id": ["B"],
        "sequence": "ACDEFGHIK",
        "count": 1
      }
    }
  ],
  "templates": [
    {
      "cif": "/path/to/template.cif",
      "chain_id": ["A", "B"],
      "template_id": ["H", "L"]
    }
  ]
}
```

Supported template inputs:

- top-level `templates`
- per-chain `templatesPath`
- `.cif`, `.mmcif`, and `.pdb` structural template files
- explicit `chain_id` / `template_id` mapping for multichain templates
- explicit `queryIndices` / `templateIndices` residue mapping when needed
- legacy `.hhr`, `.a3m`, and JSON sidecars

Run prediction with template features enabled:

```bash
protenix pred -i input.json -o output --use_template true
```

Implementation notes:

- Structural templates are converted into the same template feature contract
  that Protenix already consumes.
- Protein chains covered by top-level `templates` are skipped during automatic
  template search preprocessing.
- Homomer chains with different template-chain mappings are handled without
  entity-level template feature copying.
- Direct structural templates do not require Kalign. Legacy `.hhr` / `.a3m`
  template processing and online template hits may still require Kalign.

## Installation

Install this fork when you need the fork-specific scoring or structural-template
features.

```bash
git clone https://github.com/cytokineking/Protenix.git
cd Protenix
pip install -e .
```

For `protenix score`, also install and configure the optional
[`protenixscore`](https://github.com/cytokineking/ProtenixScore) package.

## Running Standard Protenix Prediction

The normal Protenix prediction command remains available:

```bash
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
```

For upstream usage details, see:

- [Official Protenix repository](https://github.com/bytedance/Protenix)
- [Training and Inference Instructions](docs/training_inference_instructions.md)
- [Supported Models](docs/supported_models.md)
- [Inference JSON Format](docs/infer_json_format.md)

## Checkpoint Mirror

The `protenix-v2` checkpoint is downloaded from the Hugging Face mirror at
`TMF001/pxdesign-weights` by default because the upstream ByteDance checkpoint
endpoint can be inaccessible outside China. The downloader verifies the
published SHA256 checksum for this checkpoint. Other checkpoints and common
runtime assets continue to use the upstream ByteDance URLs.

## Notes and Limitations

- Structural templates are protein-only.
- Template features are used only with `--use_template true`.
- Confidence-head scoring evaluates supplied coordinates; it does not relax or
  generate structures.
- `protenix score` depends on the separate `protenixscore` package.
- This fork does not replace upstream Protenix documentation, benchmarks, or
  citation guidance.

## License and Citation

This fork preserves the upstream Protenix license. The Protenix project,
including code and model parameters, is released under the
[Apache 2.0 License](LICENSE).

If you use this fork in research, cite the relevant upstream Protenix work as
described in the [official repository](https://github.com/bytedance/Protenix).
