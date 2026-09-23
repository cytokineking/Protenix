"""Compare every loaded v2 parameter and buffer after stock and lazy init.

Run in the qualified Protenix environment with the same checkpoint. This
constructs and loads two models sequentially, never running inference.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from collections.abc import Mapping

import torch

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from runner.inference import InferenceRunner


def _tensor_hashes(model) -> dict[str, str]:
    values = {
        **{"parameter:" + name: tensor for name, tensor in model.named_parameters()},
        **{"buffer:" + name: tensor for name, tensor in model.named_buffers()},
    }
    return {
        name: hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest() + f"|{tuple(tensor.shape)}|{tensor.dtype}"
        for name, tensor in values.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    args = parser.parse_args()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    base = {
        **deepcopy(configs_base),
        "data": deepcopy(data_configs),
        **deepcopy(inference_configs),
    }
    def deep_update(destination, updates):
        for key, value in updates.items():
            if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
                deep_update(destination[key], value)
            else:
                destination[key] = value

    deep_update(base, deepcopy(model_configs["protenix-v2"]))
    configs = parse_configs(
        base,
        arg_str=(
            f"--model_name protenix-v2 --load_checkpoint_dir {args.checkpoint_dir} "
            f"--dump_dir {args.scratch_dir} --input_json_path {args.scratch_dir / 'unused.json'}"
        ),
        fill_required_with_null=True,
    )
    results = {}
    for mode in ("off", "portable"):
        os.environ["PROTENIX_PORTABLE_OPTIMIZATIONS"] = mode
        os.environ["_PROTENIX_PORTABLE_FEATURES"] = "lazy_init"
        start = time.perf_counter()
        runner = InferenceRunner(configs)
        torch.cuda.synchronize()
        results[mode] = {
            "construct_and_load_seconds": time.perf_counter() - start,
            "tensor_hashes": _tensor_hashes(runner.model),
        }
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    before, after = results["off"]["tensor_hashes"], results["portable"]["tensor_hashes"]
    unequal = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    print(json.dumps({
        "off_seconds": results["off"]["construct_and_load_seconds"],
        "portable_seconds": results["portable"]["construct_and_load_seconds"],
        "tensors_compared": len(set(before) | set(after)),
        "unequal": unequal[:30],
        "unequal_count": len(unequal),
    }, indent=2))
    return 0 if not unequal else 1


if __name__ == "__main__":
    raise SystemExit(main())
