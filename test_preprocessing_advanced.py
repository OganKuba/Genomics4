import pytest
import pandas as pd
import numpy as np
import h5py
import tempfile
import os
import re
from pathlib import Path
from unittest.mock import patch, MagicMock
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestHDF5DataLoading:
    """Test HDF5 data loading functionality from the notebook."""
    
    def setup_method(self):
        """Set up test HDF5 file with mock data."""
        self.temp_dir = tempfile.mkdtemp()
        self.h5_file = os.path.join(self.temp_dir, "test_geuvadis_sv_expr.h5")
        
        # Create mock data
        self.samples = ['HG00001', 'HG00002', 'HG00003']
        self.genes = ['ENSG00000001.1', 'ENSG00000002.1', 'ENSG00000003.1']
        self.sv_ids = ['DEL_1_123456', 'INS_2_789012', 'DUP_3_345678']
        
        self.expr_data = np.random.rand(len(self.samples), len(self.genes)) * 10
        self.geno_data = np.random.randint(0, 3, (len(self.samples), len(self.sv_ids)))
        
        # Save to HDF5
        with h5py.File(self.h5_file, "w") as f:
            f.create_dataset("expression", data=self.expr_data)
            f.create_dataset("expr_genes", data=np.array(self.genes, dtype="S"))
            f.create_dataset("sv_gt", data=self.geno_data)
            f.create_dataset("sv_ids", data=np.array(self.sv_ids, dtype="S"))
            f.create_dataset("samples", data=np.array(self.samples, dtype="S20"))
    
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        os.rmdir(self.temp_dir)
    
    def test_hdf5_structure_validation(self):
        """Test validation of HDF5 file structure."""
        with h5py.File(self.h5_file, 'r') as f:
            # Check all required datasets exist
            required_datasets = ['expression', 'expr_genes', 'sv_gt', 'sv_ids', 'samples']
            for dataset in required_datasets:
                assert dataset in f.keys()
            
            # Check data shapes are consistent
            n_samples = len(f['samples'])
            n_genes = len(f['expr_genes'])
            n_svs = len(f['sv_ids'])
            
            assert f['expression'].shape == (n_samples, n_genes)
            assert f['sv_gt'].shape == (n_samples, n_svs)
    
    def test_expression_data_loading(self):
        """Test loading and formatting expression data."""
        with h5py.File(self.h5_file, 'r') as h5:
            expr_matrix = h5['expression'][:]
            genes = [g.decode() if hasattr(g, 'decode') else g for g in h5['expr_genes'][:]]
            samples = [s.decode() if hasattr(s, 'decode') else s for s in h5['samples'][:]]
        
        # Create expression DataFrame (transposed to genes x samples)
        expr = pd.DataFrame(expr_matrix, index=samples, columns=genes).T
        
        assert expr.shape == (len(self.genes), len(self.samples))
        assert list(expr.columns) == self.samples
        assert list(expr.index) == self.genes
        assert np.allclose(expr.values, self.expr_data.T)
    
    def test_genotype_data_loading(self):
        """Test loading genotype data from HDF5."""
        with h5py.File(self.h5_file, 'r') as h5:
            geno_matrix = h5['sv_gt'][:]
            sv_ids = [s.decode() if