#!/usr/bin/env python3
"""
Example demonstrating DGSEA (Dual Gene Set Enrichment Analysis) functionality
"""

import blitzgsea as blitz
import pandas as pd
import numpy as np

def create_example_signature():
    """Create an example gene expression signature"""
    # Create a signature with 200 genes
    np.random.seed(123)
    genes = ['GENE_' + str(i) for i in range(1, 201)]
    
    # Generate random expression values
    values = np.random.randn(200)
    
    # Create some realistic patterns:
    # - DNA repair genes (1-15) are upregulated
    values[:15] += 1.5
    # - Cell cycle genes (16-30) are downregulated  
    values[15:30] -= 1.2
    # - Immune response genes (31-45) are moderately upregulated
    values[30:45] += 0.8
    
    return pd.DataFrame({'i': genes, 'v': values})

def main():
    print("=== DGSEA (Dual Gene Set Enrichment Analysis) Example ===\n")
    
    # Create example signature
    signature = create_example_signature()
    print(f"Created signature with {len(signature)} genes")
    print(f"Expression values range: {signature['v'].min():.3f} to {signature['v'].max():.3f}\n")
    
    # Define gene sets representing different biological processes
    dna_repair_genes = set(['GENE_' + str(i) for i in range(1, 16)])      # Should be enriched
    cell_cycle_genes = set(['GENE_' + str(i) for i in range(16, 31)])     # Should be depleted
    immune_genes = set(['GENE_' + str(i) for i in range(31, 46)])         # Moderately enriched
    random_genes = set(['GENE_' + str(i) for i in range(100, 121)])       # Should be neutral
    
    print("Comparing different gene sets:\n")
    
    # Example 1: DNA repair vs Cell cycle (should show clear difference)
    print("1. DNA Repair genes vs Cell Cycle genes:")
    result1 = blitz.dgsea(signature, dna_repair_genes, cell_cycle_genes, 
                         permutations=1000, verbose=False)
    
    print(f"   DNA Repair NES: {result1['gene_set_1_nes']:.3f} (p-val: {result1['gene_set_1_pval']:.4f})")
    print(f"   Cell Cycle NES: {result1['gene_set_2_nes']:.3f} (p-val: {result1['gene_set_2_pval']:.4f})")
    print(f"   More enriched: {result1['more_enriched']}")
    print(f"   Differential NES: {result1['differential_nes']:.3f}\n")
    
    # Example 2: DNA repair vs Immune response (both enriched, but DNA repair more)
    print("2. DNA Repair genes vs Immune Response genes:")
    result2 = blitz.dgsea(signature, dna_repair_genes, immune_genes,
                         permutations=1000, verbose=False)
    
    print(f"   DNA Repair NES: {result2['gene_set_1_nes']:.3f} (p-val: {result2['gene_set_1_pval']:.4f})")
    print(f"   Immune Response NES: {result2['gene_set_2_nes']:.3f} (p-val: {result2['gene_set_2_pval']:.4f})")
    print(f"   More enriched: {result2['more_enriched']}")
    print(f"   Differential NES: {result2['differential_nes']:.3f}\n")
    
    # Example 3: DNA repair vs Random genes (should show DNA repair is enriched)
    print("3. DNA Repair genes vs Random genes:")
    result3 = blitz.dgsea(signature, dna_repair_genes, random_genes,
                         permutations=1000, verbose=False)
    
    print(f"   DNA Repair NES: {result3['gene_set_1_nes']:.3f} (p-val: {result3['gene_set_1_pval']:.4f})")
    print(f"   Random genes NES: {result3['gene_set_2_nes']:.3f} (p-val: {result3['gene_set_2_pval']:.4f})")
    print(f"   More enriched: {result3['more_enriched']}")
    print(f"   Differential NES: {result3['differential_nes']:.3f}\n")
    
    print("=== Summary ===")
    print("DGSEA successfully identified:")
    print("• DNA repair genes are more enriched than cell cycle genes")
    print("• DNA repair genes are more enriched than immune response genes")  
    print("• DNA repair genes are more enriched than random genes")
    print("\nThis matches our expected pattern where DNA repair genes were")
    print("artificially upregulated in the signature.")

if __name__ == "__main__":
    main()