"""Focused checks for the opt-in Protenix inference policy."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from protenix.inference_optimizations import (
    resolve_policy,
    skip_random_initialization,
)


class TestInferenceOptimizations(unittest.TestCase):
    def test_policy_default_and_strict_guard(self):
        config = SimpleNamespace(model_name="protenix-v2", load_strict=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_policy(config).mode, "off")
            self.assertFalse(resolve_policy(config).features)
        with patch.dict(os.environ, {"PROTENIX_PORTABLE_OPTIMIZATIONS": "portable"}):
            self.assertIn("lazy_init", resolve_policy(config).features)
            self.assertIn("release_prediction", resolve_policy(config).features)
            with self.assertRaisesRegex(ValueError, "load_strict"):
                resolve_policy(SimpleNamespace(model_name="protenix-v2", load_strict=False))
            with self.assertRaisesRegex(ValueError, "protenix-v2"):
                resolve_policy(SimpleNamespace(model_name="other", load_strict=True))
        with patch.dict(os.environ, {"PROTENIX_PORTABLE_OPTIMIZATIONS": "faster"}):
            with self.assertRaisesRegex(ValueError, "must be"):
                resolve_policy(config)

    def test_random_init_is_scoped_and_restored_after_exception(self):
        original = torch.nn.init.kaiming_uniform_
        tensor = torch.full((4, 4), 7.0)
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with skip_random_initialization() as patched:
                self.assertTrue(patched)
                torch.nn.init.kaiming_uniform_(tensor)
                self.assertTrue(torch.equal(tensor, torch.full_like(tensor, 7.0)))
                torch.nn.init.zeros_(tensor)
                self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))
                raise RuntimeError("test exception")
        self.assertIs(torch.nn.init.kaiming_uniform_, original)

if __name__ == "__main__":
    unittest.main()
