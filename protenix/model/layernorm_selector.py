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

import os

import torch


def cuda_has_blackwell_or_newer_device() -> bool:
    """Return True when any visible CUDA device has Blackwell-or-newer capability."""
    try:
        if not torch.cuda.is_available():
            return False
        for device_idx in range(torch.cuda.device_count()):
            major, _minor = torch.cuda.get_device_capability(device_idx)
            if major >= 10:
                return True
    except (AssertionError, RuntimeError):
        return False
    return False


def resolve_layernorm_type() -> str:
    """Resolve the Protenix layernorm backend, honoring explicit user overrides."""
    layernorm_type = os.getenv("LAYERNORM_TYPE")
    if layernorm_type:
        return layernorm_type
    if cuda_has_blackwell_or_newer_device():
        return "torch"
    return "fast_layernorm"
