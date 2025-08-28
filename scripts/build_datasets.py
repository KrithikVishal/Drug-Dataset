#!/usr/bin/env python3
"""
AI-augmented dataset pipeline for drug discovery.
Main build script to orchestrate data collection, processing, and feature generation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def build_dti_dataset(output_dir: Path, limit_chembl: Optional[int] = None):
    """Build Drug-Target Interaction dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Building DTI dataset...")
    
    # Import here to avoid dependency issues during help/arg parsing
    from connectors.chembl_connector import ChEMBLConnector
    from connectors.uniprot_connector import UniProtConnector
    from features.molecular_features import MolecularFeatureGenerator
    from features.protein_features import ProteinFeatureGenerator
    
    # Initialize connectors
    chembl = ChEMBLConnector()
    uniprot = UniProtConnector()
    mol_features = MolecularFeatureGenerator()
    protein_features = ProteinFeatureGenerator()
    
    # Fetch data
    dti_data = chembl.get_bioactivities(limit=limit_chembl)
    target_data = uniprot.get_target_info(dti_data['target_id'].unique())
    
    # Generate features
    mol_vectors = mol_features.generate_fingerprints(dti_data['canonical_smiles'])
    mol_descriptors = mol_features.generate_descriptors(dti_data['canonical_smiles'])
    protein_embeddings = protein_features.generate_embeddings(target_data['sequence'])
    
    # Save outputs
    dti_meta_path = output_dir / "dti_meta.csv"
    dti_vectors_path = output_dir / "dti_vectors.parquet"
    
    # Combine metadata
    meta_data = dti_data.merge(target_data, on='target_id', how='left')
    meta_data.to_csv(dti_meta_path, index=False)
    
    # Save vectors
    import pandas as pd
    vectors_df = pd.DataFrame({
        'molecule_id': dti_data['molecule_id'],
        'target_id': dti_data['target_id'],
        'mol_fingerprint': mol_vectors.tolist(),
        'rdkit_physchem': mol_descriptors.tolist(),
        'protein_embedding': protein_embeddings.tolist()
    })
    vectors_df.to_parquet(dti_vectors_path, index=False)
    
    logger.info(f"DTI dataset saved to {output_dir}")
    return dti_meta_path, dti_vectors_path


def build_admet_dataset(output_dir: Path, limit_chembl: Optional[int] = None):
    """Build ADMET dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Building ADMET dataset...")
    
    # Import here to avoid dependency issues during help/arg parsing
    from connectors.chembl_connector import ChEMBLConnector
    from features.molecular_features import MolecularFeatureGenerator
    
    chembl = ChEMBLConnector()
    mol_features = MolecularFeatureGenerator()
    
    # Fetch ADMET data
    admet_data = chembl.get_admet_data(limit=limit_chembl)
    
    # Generate molecular features
    mol_vectors = mol_features.generate_fingerprints(admet_data['canonical_smiles'])
    mol_descriptors = mol_features.generate_descriptors(admet_data['canonical_smiles'])
    
    # Save outputs
    admet_meta_path = output_dir / "admet_meta.csv"
    admet_vectors_path = output_dir / "admet_vectors.parquet"
    
    admet_data.to_csv(admet_meta_path, index=False)
    
    import pandas as pd
    vectors_df = pd.DataFrame({
        'molecule_id': admet_data['molecule_id'],
        'mol_fingerprint': mol_vectors.tolist(),
        'rdkit_physchem': mol_descriptors.tolist()
    })
    vectors_df.to_parquet(admet_vectors_path, index=False)
    
    logger.info(f"ADMET dataset saved to {output_dir}")
    return admet_meta_path, admet_vectors_path


def build_molecule_pool(output_dir: Path, limit_chembl: Optional[int] = None):
    """Build standardized molecule pool."""
    logger = logging.getLogger(__name__)
    logger.info("Building molecule pool...")
    
    # Import here to avoid dependency issues during help/arg parsing
    from connectors.chembl_connector import ChEMBLConnector
    from features.molecular_features import MolecularFeatureGenerator
    
    chembl = ChEMBLConnector()
    mol_features = MolecularFeatureGenerator()
    
    # Get diverse molecule set
    molecules = chembl.get_molecule_pool(limit=limit_chembl)
    
    # Standardize and generate features
    standardized = mol_features.standardize_molecules(molecules['canonical_smiles'])
    mol_vectors = mol_features.generate_fingerprints(standardized['canonical_smiles'])
    mol_descriptors = mol_features.generate_descriptors(standardized['canonical_smiles'])
    
    # Calculate drug-likeness metrics
    qed_scores = mol_features.calculate_qed(standardized['canonical_smiles'])
    lipinski_violations = mol_features.calculate_lipinski_violations(standardized['canonical_smiles'])
    
    # Save outputs
    pool_meta_path = output_dir / "molecule_pool_meta.csv"
    pool_vectors_path = output_dir / "molecule_pool_vectors.parquet"
    
    import pandas as pd
    meta_df = pd.concat([molecules, standardized, qed_scores, lipinski_violations], axis=1)
    meta_df.to_csv(pool_meta_path, index=False)
    
    vectors_df = pd.DataFrame({
        'molecule_id': molecules['molecule_id'],
        'mol_fingerprint': mol_vectors.tolist(),
        'rdkit_physchem': mol_descriptors.tolist()
    })
    vectors_df.to_parquet(pool_vectors_path, index=False)
    
    logger.info(f"Molecule pool saved to {output_dir}")
    return pool_meta_path, pool_vectors_path


def main():
    """Main entry point for dataset building."""
    parser = argparse.ArgumentParser(description="Build AI-augmented drug discovery datasets")
    parser.add_argument("--all", action="store_true", help="Build all datasets")
    parser.add_argument("--dti", action="store_true", help="Build DTI dataset")
    parser.add_argument("--admet", action="store_true", help="Build ADMET dataset")
    parser.add_argument("--molecule-pool", action="store_true", help="Build molecule pool")
    parser.add_argument("--chembl", action="store_true", help="Use ChEMBL connector")
    parser.add_argument("--uniprot", action="store_true", help="Use UniProt connector")
    parser.add_argument("--limit-chembl", type=int, help="Limit ChEMBL records")
    parser.add_argument("--out", type=str, default="data/", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting dataset pipeline...")
    logger.info(f"Output directory: {output_dir}")
    
    built_files = []
    
    try:
        if args.all or args.dti:
            dti_files = build_dti_dataset(output_dir, args.limit_chembl)
            built_files.extend(dti_files)
        
        if args.all or args.admet:
            admet_files = build_admet_dataset(output_dir, args.limit_chembl)
            built_files.extend(admet_files)
        
        if args.all or args.molecule_pool:
            pool_files = build_molecule_pool(output_dir, args.limit_chembl)
            built_files.extend(pool_files)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Built files: {built_files}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()