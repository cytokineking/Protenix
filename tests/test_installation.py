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

# Copyright 2025 Shad Nygren, Virtual Hipster Corporation
# Contributed to the Protenix project under the Apache License 2.0

"""
Test suite for installation and dependency compatibility issues.
Related to GitHub issues #182 (pip install error)
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestInstallation(unittest.TestCase):
    """Test that package dependencies are compatible"""

    def test_explicit_mps_requires_metal(self):
        from runner.inference import inference_device

        with patch.dict(os.environ, PROTENIX_DEVICE="mps"), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "PROTENIX_DEVICE"):
                inference_device()

    def test_auto_selects_cuda_before_mps(self):
        from runner.inference import inference_device

        with patch.dict(os.environ, PROTENIX_DEVICE="auto"), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(inference_device().type, "cuda")

    def test_device_kernel_selection(self):
        import torch

        from runner.inference import update_gpu_compatible_configs

        for device, capability, expected_dtype, expected_kernel, fusion in [
            ("mps", None, "fp32", "torch", False),
            ("cpu", None, "fp32", "torch", False),
            ("cuda", (7, 0), "fp32", "torch", True),
            ("cuda", (9, 0), "bf16", "cuequivariance", True),
        ]:
            with self.subTest(device=device, capability=capability), patch(
                "torch.cuda.get_device_capability", return_value=capability
            ):
                configs = SimpleNamespace(dtype="bf16", triangle_attention="cuequivariance",
                                          triangle_multiplicative="cuequivariance",
                                          enable_efficient_fusion=True, enable_tf32=True)
                update_gpu_compatible_configs(configs, torch.device(device))
                self.assertEqual(configs.dtype, expected_dtype)
                self.assertEqual(configs.triangle_attention, expected_kernel)
                self.assertEqual(configs.triangle_multiplicative, expected_kernel)
                self.assertEqual(configs.enable_efficient_fusion, fusion)
                self.assertEqual(configs.enable_tf32, fusion)

    def test_python_version(self):
        """Verify Python version is compatible (3.11+)"""
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(
            version_info.minor, 11, "Protenix requires Python 3.11 or higher"
        )

    def test_cuda_accelerators_are_backend_specific_extras(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text()
        setup_py = (root / "setup.py").read_text()

        self.assertNotIn("cuequivariance-ops-torch-cu12", requirements)
        self.assertNotIn("cuequivariance-ops-torch-cu13", requirements)
        self.assertIn(
            '"cu12": ["cuequivariance-ops-torch-cu12==0.10.0"]', setup_py
        )
        self.assertIn(
            '"cu13": ["cuequivariance-ops-torch-cu13==0.10.0"]', setup_py
        )

    def test_deepspeed_pydantic_compatibility(self):
        """
        Test for issue #182: DeepSpeed and Pydantic compatibility
        The error "json_schema_input_schema" indicates version mismatch
        """
        try:
            # Try to import DeepSpeed - this will fail if incompatible versions
            deepspeed_spec = importlib.util.find_spec("deepspeed")
            if deepspeed_spec is not None:
                import deepspeed  # noqa: F401
                import pydantic  # noqa: F401

                # This import caused the original error in issue #182
                try:
                    from deepspeed.runtime.config import DeepSpeedConfig  # noqa: F401

                    # If we get here, the versions are compatible
                    self.assertTrue(True)
                except TypeError as e:
                    if "json_schema_input_schema" in str(e):
                        self.fail(
                            "DeepSpeed/Pydantic compatibility issue detected. "
                            "This is the bug from issue #182. "
                            "Consider pinning pydantic<2.0 or updating DeepSpeed."
                        )
                    else:
                        raise
        except ImportError:
            self.skipTest("DeepSpeed not installed - skipping compatibility test")

    def test_required_packages_importable(self):
        """Verify core packages can be imported"""
        required_packages = [
            "torch",
            "numpy",
            "scipy",
            "pandas",
            "ml_collections",
        ]

        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is None:
                self.skipTest(
                    f"{package} not installed - this test documents missing dependencies"
                )

    def test_protenix_imports(self):
        """Test that protenix package can be imported when installed"""
        try:
            import protenix  # noqa: F401

            self.assertTrue(True)
        except ImportError:
            self.skipTest(
                "Protenix not installed - this documents the need for installation"
            )

    def test_cuda_availability(self):
        """Document CUDA availability for GPU acceleration"""
        try:
            import torch

            if torch.cuda.is_available():
                self.assertTrue(True, "CUDA is available")
            else:
                # Not a failure, just documentation
                print("Note: CUDA not available, will use CPU")
        except ImportError:
            self.skipTest("PyTorch not installed")


if __name__ == "__main__":
    unittest.main()  # Test signed commit
