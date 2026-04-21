"""
Basic tests for the dataset pipeline components.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from connectors.chembl_connector import ChEMBLConnector
        from connectors.uniprot_connector import UniProtConnector
        from features.molecular_features import MolecularFeatureGenerator
        from features.protein_features import ProteinFeatureGenerator
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_molecular_features():
    """Test molecular feature generation."""
    from features.molecular_features import MolecularFeatureGenerator
    
    generator = MolecularFeatureGenerator()
    
    # Test with simple molecules
    smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']  # ethanol, acetic acid, benzene
    
    # Test fingerprint generation
    fingerprints = generator.generate_fingerprints(smiles)
    assert fingerprints.shape[0] == len(smiles)
    assert fingerprints.shape[1] == 1024  # Default Morgan fingerprint size
    
    # Test descriptor generation
    descriptors = generator.generate_descriptors(smiles)
    assert descriptors.shape[0] == len(smiles)
    assert descriptors.shape[1] > 0

def test_protein_features():
    """Test protein feature generation."""
    from features.protein_features import ProteinFeatureGenerator
    
    generator = ProteinFeatureGenerator()
    
    # Test with simple protein sequences
    sequences = ['ACDEFGHIKLMNPQRSTVWY', 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG']
    
    # Test embedding generation
    embeddings = generator.generate_embeddings(sequences, method="physicochemical")
    assert embeddings.shape[0] == len(sequences)
    assert embeddings.shape[1] > 0

def test_build_script_args():
    """Test that build script can parse arguments."""
    import subprocess
    import sys
    
    # Test help message
    result = subprocess.run([
        sys.executable, 'scripts/build_datasets.py', '--help'
    ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    assert result.returncode == 0
    assert 'Build AI-augmented drug discovery datasets' in result.stdout

if __name__ == "__main__":
    pytest.main([__file__])