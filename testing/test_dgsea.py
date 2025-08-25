#!/usr/bin/env python3
"""
Unit tests for DGSEA (Dual Gene Set Enrichment Analysis) functionality
"""

import unittest
import blitzgsea as blitz
import pandas as pd
import numpy as np


class TestDGSEA(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Set seed for reproducible tests
        np.random.seed(42)
        
        # Create a test signature with 100 genes
        self.genes = ['GENE_' + str(i) for i in range(1, 101)]
        self.values = np.random.randn(100)
        
        # Make first 20 genes highly enriched (positive values)
        self.values[:20] += 2.0
        # Make genes 21-40 depleted (negative values)
        self.values[20:40] -= 1.5
        
        self.signature = pd.DataFrame({'i': self.genes, 'v': self.values})
        
        # Create test gene sets
        self.enriched_set = set(['GENE_' + str(i) for i in range(1, 21)])  # Top 20 genes
        self.depleted_set = set(['GENE_' + str(i) for i in range(21, 41)])  # Depleted genes
        self.random_set = set(['GENE_' + str(i) for i in range(50, 70)])   # Random genes
    
    def test_basic_functionality(self):
        """Test basic DGSEA functionality"""
        result = blitz.dgsea(
            self.signature, 
            self.enriched_set, 
            self.depleted_set, 
            permutations=500,
            seed=42
        )
        
        # Check return type and structure
        self.assertIsInstance(result, dict)
        expected_keys = {
            'gene_set_1_es', 'gene_set_2_es', 'gene_set_1_nes', 'gene_set_2_nes',
            'gene_set_1_pval', 'gene_set_2_pval', 'differential_es', 
            'differential_nes', 'more_enriched'
        }
        self.assertEqual(set(result.keys()), expected_keys)
        
        # Check that enriched set has positive NES and depleted set has negative NES
        self.assertGreater(result['gene_set_1_nes'], 0, "Enriched set should have positive NES")
        self.assertLess(result['gene_set_2_nes'], 0, "Depleted set should have negative NES")
        
        # Check that enriched set is identified as more enriched
        self.assertEqual(result['more_enriched'], 'gene_set_1')
        
        # Check statistical significance
        self.assertLess(result['gene_set_1_pval'], 0.05, "Enriched set should be significant")
        self.assertLess(result['gene_set_2_pval'], 0.05, "Depleted set should be significant")
    
    def test_identical_gene_sets(self):
        """Test DGSEA with identical gene sets"""
        result = blitz.dgsea(
            self.signature,
            self.enriched_set,
            self.enriched_set,  # Same set
            permutations=100,
            seed=42
        )
        
        # ES should be identical for identical sets
        self.assertAlmostEqual(
            result['gene_set_1_es'], 
            result['gene_set_2_es'], 
            places=10,
            msg="Identical sets should have identical enrichment scores"
        )
        
        # Differential ES should be zero
        self.assertAlmostEqual(
            result['differential_es'], 
            0.0, 
            places=10,
            msg="Differential ES should be zero for identical sets"
        )
    
    def test_small_gene_sets(self):
        """Test DGSEA with small gene sets"""
        small_set1 = set(['GENE_1', 'GENE_2'])
        small_set2 = set(['GENE_99', 'GENE_100'])
        
        result = blitz.dgsea(
            self.signature,
            small_set1,
            small_set2,
            permutations=100,
            seed=42
        )
        
        # Should work with small sets
        self.assertIsInstance(result, dict)
        self.assertIn('gene_set_1_es', result)
        self.assertIn('gene_set_2_es', result)
    
    def test_no_overlap_error(self):
        """Test error handling when gene sets don't overlap with signature"""
        signature = pd.DataFrame({'i': ['A', 'B', 'C'], 'v': [1, 0, -1]})
        no_overlap_set = set(['X', 'Y', 'Z'])
        valid_set = set(['A', 'B'])
        
        # Should raise ValueError when gene set has no overlap
        with self.assertRaises(ValueError) as context:
            blitz.dgsea(signature, no_overlap_set, valid_set, permutations=10)
        
        self.assertIn("no genes in common", str(context.exception))
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test with different seeds
        result1 = blitz.dgsea(
            self.signature, 
            self.enriched_set, 
            self.depleted_set, 
            permutations=100, 
            seed=42
        )
        
        result2 = blitz.dgsea(
            self.signature, 
            self.enriched_set, 
            self.depleted_set, 
            permutations=100, 
            seed=42
        )
        
        # Results should be identical with same seed
        self.assertAlmostEqual(result1['gene_set_1_es'], result2['gene_set_1_es'], places=10)
        self.assertAlmostEqual(result1['gene_set_2_es'], result2['gene_set_2_es'], places=10)
    
    def test_differential_scores(self):
        """Test differential score calculations"""
        result = blitz.dgsea(
            self.signature,
            self.enriched_set,
            self.depleted_set,
            permutations=100,
            seed=42
        )
        
        # Verify differential calculations
        expected_diff_es = result['gene_set_1_es'] - result['gene_set_2_es']
        expected_diff_nes = result['gene_set_1_nes'] - result['gene_set_2_nes']
        
        self.assertAlmostEqual(result['differential_es'], expected_diff_es, places=10)
        self.assertAlmostEqual(result['differential_nes'], expected_diff_nes, places=10)


if __name__ == '__main__':
    unittest.main()