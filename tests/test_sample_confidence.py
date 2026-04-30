import unittest
import torch
from protenix.model.sample_confidence import calculate_chain_pair_pae

class TestSampleConfidence(unittest.TestCase):
    def test_calculate_chain_pair_pae(self):
        # Setup inputs
        N_samples = 2
        N_token = 4
        N_chain = 2
        
        # Asymmetric IDs: two chains, 2 tokens each
        asym_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        
        # All tokens have frames
        token_has_frame = torch.tensor([True, True, True, False], dtype=torch.bool)
        
        # Token pair PAE: [N_samples, N_token, N_token]
        # Sample 0
        token_pair_pae_s0 = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ])
        # Sample 1
        token_pair_pae_s1 = torch.tensor([
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 2.0, 5.0, 4.0],
            [4.0, 5.0, 2.0, 3.0],
            [5.0, 4.0, 3.0, 2.0],
        ])
        token_pair_pae = torch.stack([token_pair_pae_s0, token_pair_pae_s1])
        
        # Expected output shapes
        # chain_pair_pae_mean: [N_samples, N_chain, N_chain]
        # chain_pair_pae_min: [N_samples, N_chain, N_chain]
        
        result = calculate_chain_pair_pae(
            token_pair_pae=token_pair_pae,
            asym_id=asym_id,
            token_has_frame=token_has_frame,
        )
        
        self.assertIn("chain_pair_pae_mean", result)
        self.assertIn("chain_pair_pae_min", result)
        
        chain_pair_pae_mean = result["chain_pair_pae_mean"]
        chain_pair_pae_min = result["chain_pair_pae_min"]
        
        self.assertEqual(chain_pair_pae_mean.shape, (N_samples, N_chain, N_chain))
        self.assertEqual(chain_pair_pae_min.shape, (N_samples, N_chain, N_chain))
        
        # Check chain 1 vs chain 1 (Sample 0)
        # For chain 1 vs chain 1, the valid tokens are only token 2 (token 3 has frame=False)
        # token_pair_pae_s0[2, 2] = 1.0
        # min = 1.0, mean = 1.0
        self.assertTrue(torch.allclose(chain_pair_pae_min[0, 1, 1], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(chain_pair_pae_mean[0, 1, 1], torch.tensor(1.0)))
        
        # Check chain 0 vs chain 1 (Sample 0)
        # For chain 0 vs chain 1, valid tokens for chain 0 are 0, 1. For chain 1 is 2.
        # sub_pae for sample 0 is token_pair_pae_s0[0:2, 2] -> [3.0, 4.0]
        # min = min(3.0, 4.0) = 3.0
        # mean = (3.0 + 4.0) / 2 = 3.5
        self.assertTrue(torch.allclose(chain_pair_pae_min[0, 0, 1], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(chain_pair_pae_mean[0, 0, 1], torch.tensor(3.5)))

if __name__ == "__main__":
    unittest.main()
