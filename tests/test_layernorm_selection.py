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
import unittest
from unittest.mock import patch

from protenix.model import layernorm_selector


class TestLayerNormSelection(unittest.TestCase):
    def test_explicit_layernorm_type_wins(self) -> None:
        with patch.dict(os.environ, {"LAYERNORM_TYPE": "fast_layernorm"}), patch.object(
            layernorm_selector.torch.cuda, "is_available", return_value=True
        ), patch.object(
            layernorm_selector.torch.cuda, "device_count", return_value=1
        ), patch.object(
            layernorm_selector.torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            self.assertEqual(
                layernorm_selector.resolve_layernorm_type(), "fast_layernorm"
            )

    def test_blackwell_defaults_to_torch_layernorm(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            layernorm_selector.torch.cuda, "is_available", return_value=True
        ), patch.object(
            layernorm_selector.torch.cuda, "device_count", return_value=1
        ), patch.object(
            layernorm_selector.torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            self.assertEqual(layernorm_selector.resolve_layernorm_type(), "torch")

    def test_non_blackwell_cuda_defaults_to_fast_layernorm(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            layernorm_selector.torch.cuda, "is_available", return_value=True
        ), patch.object(
            layernorm_selector.torch.cuda, "device_count", return_value=1
        ), patch.object(
            layernorm_selector.torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            self.assertEqual(
                layernorm_selector.resolve_layernorm_type(), "fast_layernorm"
            )

    def test_no_cuda_defaults_to_fast_layernorm(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            layernorm_selector.torch.cuda, "is_available", return_value=False
        ):
            self.assertEqual(
                layernorm_selector.resolve_layernorm_type(), "fast_layernorm"
            )


if __name__ == "__main__":
    unittest.main()
