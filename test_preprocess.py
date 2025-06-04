import pytest
import pandas as pd
import numpy as np
import h5py
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path to import preprocess module
sys.path.insert(0, str(Path(__file__).parent))

class TestGeneCountProcessing:
    """Test gene count data loading and TPM conversion."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock gene count data
        self.mock_gene_data = pd.DataFrame({
            'Geneid': ['ENSG00000001.1', 'ENSG00000002.1', 'ENSG00000003.1'],
            'Chr': ['chr1', 'chr2', 'chr3'],
            'Start': [100, 200, 300],
            'End': [1100, 1200, 1300],
            'Strand': ['+', '-', '+'],
            'Length': [1000, 1000, 1000],
            'HG00001.bam': [100, 200, 50],
            'HG00002.bam': [150, 100, 75],
            'HG00003.bam': [200, 300, 100]
        })
        
        # Create temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_counts.txt")
        self.mock_gene_data.to_csv(self.temp_file, sep='\t', index=False)
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_load_gene_counts(self):
        """Test loading gene count data from file."""
        df = pd.read_csv(self.temp_file, sep="\t", comment="#")
        
        assert len(df) == 3
        assert 'Geneid' in df.columns
        assert 'Length' in df.columns
        assert all(col in df.columns for col in ['HG00001.bam', 'HG00002.bam', 'HG00003.bam'])
    
    def test_expression_matrix_creation(self):
        """Test creation of expression matrix from gene counts."""
        df = pd.read_csv(self.temp_file, sep="\t", comment="#")
        
        # Extract expression data (similar to preprocess.py logic)
        expr_raw = df.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
        expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
        
        assert expr_raw.shape == (3, 3)  # 3 samples, 3 genes
        assert all(idx.startswith('HG') for idx in expr_raw.index)
        assert expr_raw.loc['HG00001', 'ENSG00000001.1'] == 100
    
    def test_tpm_conversion(self):
        """Test conversion from raw counts to TPM (Transcripts Per Million)."""
        df = pd.read_csv(self.temp_file, sep="\t", comment="#")
        
        gene_len_kb = df["Length"] / 1e3
        expr_raw = df.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
        expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
        
        # Calculate RPK (Reads per Kilobase)
        rpk = expr_raw.div(gene_len_kb.values, axis=1)
        
        # Calculate TPM
        tpm = rpk.div(rpk.sum(axis=1), axis=0) * 1e6
        
        # Test TPM properties
        assert np.allclose(tpm.sum(axis=1), 1e6)  # Each sample sums to 1M
        assert tpm.shape == expr_raw.shape
        assert (tpm >= 0).all().all()  # All values non-negative
    
    def test_log_transformation(self):
        """Test log2 transformation of TPM values."""
        df = pd.read_csv(self.temp_file, sep="\t", comment="#")
        
        gene_len_kb = df["Length"] / 1e3
        expr_raw = df.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
        expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
        rpk = expr_raw.div(gene_len_kb.values, axis=1)
        tpm = rpk.div(rpk.sum(axis=1), axis=0) * 1e6
        
        # Log transform
        expr = np.log2(tpm + 1)
        
        assert expr.shape == tpm.shape
        assert (expr >= 0).all().all()  # log2(x+1) >= 0 for x >= 0
        assert np.isfinite(expr).all().all()  # No infinite values
    
    def test_gene_filtering(self):
        """Test filtering genes based on expression threshold."""
        df = pd.read_csv(self.temp_file, sep="\t", comment="#")
        
        gene_len_kb = df["Length"] / 1e3
        expr_raw = df.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
        expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
        rpk = expr_raw.div(gene_len_kb.values, axis=1)
        tpm = rpk.div(rpk.sum(axis=1), axis=0) * 1e6
        
        # Apply filtering (genes with TPM > 1 in at least 20% of samples)
        mask = (tpm > 1).sum(axis=0) >= int(0.2 * tpm.shape[0])
        tpm_filtered = tpm.loc[:, mask]
        
        assert tpm_filtered.shape[0] == tpm.shape[0]  # Same number of samples
        assert tpm_filtered.shape[1] <= tpm.shape[1]  # Fewer or equal genes


class TestVCFProcessing:
    """Test VCF file processing for structural variants."""
    
    def test_mock_vcf_processing(self):
        """Test VCF processing with mock data."""
        # Mock VCF data structure
        mock_samples = ['HG00001', 'HG00002', 'HG00003']
        mock_variants = [
            {'ID': 'DEL_1_123456', 'genotypes': [(0, 1, False), (1, 1, False), (0, 0, False)]},
            {'ID': 'INS_2_789012', 'genotypes': [(1, 0, False), (0, 0, False), (0, 1, False)]},
            {'ID': 'DUP_3_345678', 'genotypes': [(0, 0, False), (1, 1, False), (1, 0, False)]}
        ]
        
        # Process genotypes (similar to preprocess.py logic)
        geno = []
        sv_ids = []
        
        for var in mock_variants:
            geno.append([(a+b if a>=0 and b>=0 else 0) for a,b,*_ in var['genotypes']])
            sv_ids.append(var['ID'])
        
        geno_df = pd.DataFrame(np.array(geno).T, index=mock_samples, columns=sv_ids)
        
        assert geno_df.shape == (3, 3)  # 3 samples, 3 SVs
        assert all(sample in mock_samples for sample in geno_df.index)
        assert all(sv_id in sv_ids for sv_id in geno_df.columns)
        assert set(geno_df.values.flatten()) <= {0, 1, 2}  # Only valid genotype values
    
    def test_genotype_encoding(self):
        """Test proper encoding of VCF genotypes to numeric format."""
        # Test different genotype scenarios
        test_cases = [
            ((0, 0, False), 0),  # Homozygous reference
            ((0, 1, False), 1),  # Heterozygous
            ((1, 1, False), 2),  # Homozygous alternate
            ((-1, -1, False), 0), # Missing data should become 0
            ((0, -1, False), 0)   # Partial missing data should become 0
        ]
        
        for genotype, expected in test_cases:
            a, b = genotype[0], genotype[1]
            result = a + b if a >= 0 and b >= 0 else 0
            assert result == expected


class TestDataSynchronization:
    """Test synchronization between expression and genotype data."""
    
    def setup_method(self):
        """Set up test data for synchronization tests."""
        # Expression data
        self.expr_samples = ['HG00001', 'HG00002', 'HG00003', 'HG00004']
        self.expr_genes = ['ENSG001', 'ENSG002', 'ENSG003']
        self.expr_data = pd.DataFrame(
            np.random.rand(len(self.expr_samples), len(self.expr_genes)),
            index=self.expr_samples,
            columns=self.expr_genes
        )
        
        # Genotype data (with some overlap)
        self.geno_samples = ['HG00002', 'HG00003', 'HG00004', 'HG00005']
        self.sv_ids = ['DEL_1', 'INS_2', 'DUP_3']
        self.geno_data = pd.DataFrame(
            np.random.randint(0, 3, size=(len(self.geno_samples), len(self.sv_ids))),
            index=self.geno_samples,
            columns=self.sv_ids
        )
    
    def test_sample_intersection(self):
        """Test finding common samples between expression and genotype data."""
        common = self.expr_data.index.intersection(self.geno_data.index)
        
        expected_common = {'HG00002', 'HG00003', 'HG00004'}
        assert set(common) == expected_common
        assert len(common) == 3
    
    def test_data_synchronization(self):
        """Test synchronizing expression and genotype data to common samples."""
        common = self.expr_data.index.intersection(self.geno_data.index)
        
        expr_sync = self.expr_data.loc[common].sort_index()
        geno_sync = self.geno_data.loc[common].sort_index()
        
        assert expr_sync.shape[0] == geno_sync.shape[0]
        assert all(expr_sync.index == geno_sync.index)
        assert expr_sync.shape[0] == len(common)
    
    def test_empty_intersection(self):
        """Test handling case with no common samples."""
        # Create data with no overlap
        expr_different = pd.DataFrame(
            np.random.rand(2, 3),
            index=['Sample1', 'Sample2'],
            columns=['Gene1', 'Gene2', 'Gene3']
        )
        
        common = expr_different.index.intersection(self.geno_data.index)
        assert len(common) == 0


class TestHDF5Persistence:
    """Test HDF5 data saving and loading functionality."""
    
    def setup_method(self):
        """Set up test data for HDF5 tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.h5_file = os.path.join(self.temp_dir, "test_data.h5")
        
        # Sample data
        self.samples = ['HG00001', 'HG00002', 'HG00003']
        self.genes = ['ENSG001', 'ENSG002', 'ENSG003']
        self.sv_ids = ['DEL_1', 'INS_2']
        
        self.expr_data = np.random.rand(len(self.samples), len(self.genes))
        self.geno_data = np.random.randint(0, 3, (len(self.samples), len(self.sv_ids)))
    
    def teardown_method(self):
        """Clean up after tests."""
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        os.rmdir(self.temp_dir)
    
    def test_hdf5_save_and_load(self):
        """Test saving and loading data to/from HDF5 format."""
        # Save data
        with h5py.File(self.h5_file, "w") as f:
            f.create_dataset("expression", data=self.expr_data, compression="gzip")
            f.create_dataset("expr_genes", data=np.array(self.genes, dtype="S"))
            f.create_dataset("sv_gt", data=self.geno_data, compression="gzip")
            f.create_dataset("sv_ids", data=np.array(self.sv_ids, dtype="S"))
            f.create_dataset("samples", data=np.array(self.samples, dtype="S20"))
        
        # Load and verify data
        with h5py.File(self.h5_file, "r") as f:
            loaded_expr = f["expression"][:]
            loaded_genes = [g.decode() for g in f["expr_genes"][:]]
            loaded_geno = f["sv_gt"][:]
            loaded_sv_ids = [s.decode() for s in f["sv_ids"][:]]
            loaded_samples = [s.decode() for s in f["samples"][:]]
        
        np.testing.assert_array_equal(loaded_expr, self.expr_data)
        assert loaded_genes == self.genes
        np.testing.assert_array_equal(loaded_geno, self.geno_data)
        assert loaded_sv_ids == self.sv_ids
        assert loaded_samples == self.samples
    
    def test_hdf5_compression(self):
        """Test that compression reduces file size."""
        # Save without compression
        h5_uncompressed = os.path.join(self.temp_dir, "uncompressed.h5")
        with h5py.File(h5_uncompressed, "w") as f:
            f.create_dataset("data", data=self.expr_data)
        
        # Save with compression
        h5_compressed = os.path.join(self.temp_dir, "compressed.h5")
        with h5py.File(h5_compressed, "w") as f:
            f.create_dataset("data", data=self.expr_data, compression="gzip")
        
        size_uncompressed = os.path.getsize(h5_uncompressed)
        size_compressed = os.path.getsize(h5_compressed)
        
        # Compressed should be smaller or equal (for small test data, might be equal)
        assert size_compressed <= size_uncompressed
        
        # Clean up
        os.remove(h5_uncompressed)
        os.remove(h5_compressed)
    
    def test_hdf5_data_integrity(self):
        """Test data integrity after HDF5 round-trip."""
        # Create data with specific patterns for integrity checking
        test_expr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        test_geno = np.array([[0, 1], [1, 2], [2, 0]])
        
        # Save
        with h5py.File(self.h5_file, "w") as f:
            f.create_dataset("expression", data=test_expr)
            f.create_dataset("sv_gt", data=test_geno)
        
        # Load
        with h5py.File(self.h5_file, "r") as f:
            loaded_expr = f["expression"][:]
            loaded_geno = f["sv_gt"][:]
        
        np.testing.assert_array_equal(loaded_expr, test_expr)
        np.testing.assert_array_equal(loaded_geno, test_geno)


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_expression_data_validation(self):
        """Test validation of expression data properties."""
        # Valid expression data
        valid_expr = pd.DataFrame({
            'Sample1': [1.0, 2.0, 3.0],
            'Sample2': [2.0, 3.0, 4.0],
            'Sample3': [0.5, 1.5, 2.5]
        })
        
        # Check for finite values
        assert np.isfinite(valid_expr).all().all()
        
        # Check for non-negative values (after log2(x+1) transformation)
        log_expr = np.log2(valid_expr + 1)
        assert (log_expr >= 0).all().all()
    
    def test_genotype_data_validation(self):
        """Test validation of genotype data."""
        # Valid genotype data (0, 1, 2)
        valid_geno = pd.DataFrame({
            'Sample1': [0, 1, 2],
            'Sample2': [1, 2, 0],
            'Sample3': [2, 0, 1]
        })
        
        # Check genotype values are in valid range
        valid_values = {0, 1, 2}
        assert set(valid_geno.values.flatten()) <= valid_values
    
    def test_sample_name_consistency(self):
        """Test consistency of sample naming convention."""
        sample_names = ['HG00001', 'HG00002', 'HG00003', 'HG12345']
        
        # Check HG pattern
        pattern_check = [name.startswith('HG') and len(name) == 7 for name in sample_names]
        assert all(pattern_check)
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Data with NaN values
        data_with_nan = pd.DataFrame({
            'Sample1': [1.0, np.nan, 3.0],
            'Sample2': [2.0, 3.0, np.nan],
            'Sample3': [0.5, 1.5, 2.5]
        })
        
        # Count missing values
        missing_count = data_with_nan.isna().sum().sum()
        assert missing_count == 2
        
        # Check mask creation for valid data
        mask = ~data_with_nan.isna()
        assert mask.sum().sum() == 7  # 9 total - 2 missing


class TestIntegrationScenarios:
    """Integration tests that test the full preprocessing pipeline."""
    
    def test_end_to_end_pipeline_mock(self):
        """Test complete preprocessing pipeline with mock data."""
        # This test simulates the main workflow without actual file I/O
        
        # 1. Mock gene count data processing
        mock_counts = pd.DataFrame({
            'Geneid': ['ENSG001', 'ENSG002'],
            'Chr': ['chr1', 'chr2'],
            'Start': [100, 200],
            'End': [200, 300],
            'Strand': ['+', '+'],
            'Length': [100, 100],
            'HG00001.bam': [50, 100],
            'HG00002.bam': [75, 150]
        })
        
        # Process expression data
        gene_len_kb = mock_counts["Length"] / 1e3
        expr_raw = mock_counts.set_index("Geneid").drop(columns=["Chr","Start","End","Strand","Length"]).T
        expr_raw.index = expr_raw.index.str.extract(r'(HG\d{5})')[0]
        rpk = expr_raw.div(gene_len_kb.values, axis=1)
        tpm = rpk.div(rpk.sum(axis=1), axis=0) * 1e6
        expr = np.log2(tpm + 1)
        
        # 2. Mock genotype data
        samples = ['HG00001', 'HG00002']
        sv_ids = ['DEL_1', 'INS_2']
        geno = pd.DataFrame(
            [[0, 1], [1, 2]],
            index=samples,
            columns=sv_ids
        )
        
        # 3. Synchronization
        common = expr.index.intersection(geno.index)
        expr_sync = expr.loc[common].sort_index()
        geno_sync = geno.loc[common].sort_index()
        
        # 4. Validate final data
        assert expr_sync.shape[0] == geno_sync.shape[0]
        assert len(common) == 2
        assert all(expr_sync.index == geno_sync.index)
        assert np.isfinite(expr_sync).all().all()
        assert set(geno_sync.values.flatten()) <= {0, 1, 2}
    
    def test_pipeline_with_filtering(self):
        """Test pipeline including gene filtering step."""
        # Create data where some genes should be filtered out
        mock_tpm = pd.DataFrame({
            'HG00001': [0.5, 2.0, 10.0],  # Gene 1: low expression
            'HG00002': [0.8, 3.0, 15.0],  # Gene 2: medium expression  
            'HG00003': [0.2, 1.5, 20.0]   # Gene 3: high expression
        }, index=['Gene1', 'Gene2', 'Gene3']).T
        
        # Apply filtering (TPM > 1 in at least 20% of samples)
        threshold = 1.0
        min_samples = int(0.2 * mock_tpm.shape[0])  # At least 1 sample (20% of 3)
        
        mask = (mock_tpm > threshold).sum(axis=0) >= min_samples
        filtered_tpm = mock_tpm.loc[:, mask]
        
        # Gene1 should be filtered out (never > 1.0)
        # Gene2 and Gene3 should remain
        assert 'Gene1' not in filtered_tpm.columns
        assert 'Gene2' in filtered_tpm.columns
        assert 'Gene3' in filtered_tpm.columns
        assert filtered_tpm.shape == (3, 2)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])