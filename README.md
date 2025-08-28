# Drug-Dataset

AI-augmented pipeline to collect, standardize, and vectorize datasets for drug discovery tasks with real-time friendly schemas.

## Overview

This repository provides a comprehensive pipeline for building machine learning-ready datasets for drug discovery research. The pipeline integrates multiple data sources (ChEMBL, UniProt) and generates standardized molecular and protein features for downstream AI/ML applications.

## Features

- **Multi-source data integration**: ChEMBL bioactivities, UniProt protein data
- **Molecular feature generation**: Morgan fingerprints, RDKit descriptors, QED scores
- **Protein feature generation**: Sequence embeddings, physicochemical properties
- **Real-time friendly schemas**: Separate metadata (CSV) and vectors (Parquet)
- **Automated workflows**: GitHub Actions for continuous dataset building
- **Quality control**: Standardization, validation, and provenance tracking

## Quick Start

### Installation

```bash
git clone https://github.com/KrithikVishal/Drug-Dataset.git
cd Drug-Dataset
pip install -r requirements.txt
```

### Build Datasets

```bash
# Build all datasets with default settings
python scripts/build_datasets.py --all --chembl --uniprot --limit-chembl 1000 --out data/

# Build specific datasets
python scripts/build_datasets.py --dti --chembl --uniprot --out data/
python scripts/build_datasets.py --admet --chembl --out data/
python scripts/build_datasets.py --molecule-pool --chembl --out data/
```

### Run Tests

```bash
pytest tests/ -v
```

## Dataset Schemas

Each task emits a thin CSV + a vector Parquet:
- CSV: compact metadata, IDs, quality flags, provenance, `scaffold_id`, `split`.
- Parquet: array columns for vectors (`mol_fingerprint`, `rdkit_physchem`, optional `protein_embedding`).

### 1) Drug-Target Interactions (DTI)
- **dti_meta.csv**
  - molecule_id, canonical_smiles, inchi_key
  - is_neutralized, stereo_flag, salt_removed, standardized
  - scaffold_id, duplicate_group
  - target_id, uniprot_id, organism, target_class
  - assay_id, assay_type, endpoint_type (Ki/Kd/IC50/EC50)
  - value (nM), qualifier (>, <, =), units (nM)
  - pH, temperature, assay_confidence
  - source, citation, year, split
- **dti_vectors.parquet**
  - molecule_id, target_id
  - mol_fingerprint (int8[1024])
  - rdkit_physchem (float32[128])
  - protein_embedding (float32[N] or null)

### 2) ADMET Properties
- **admet_meta.csv**
  - molecule_id, canonical_smiles, inchi_key
  - standardization flags
  - endpoints (e.g., hepatotox_label, hERG_label, caco2_perm, logS, clint)
  - assay meta: assay_id, protocol, units, batch_id, lab
  - quality: qc_pass, replicates, stdev, outlier_flag
  - source, citation, year, split, scaffold_id, duplicate_group
- **admet_vectors.parquet**
  - molecule_id, mol_fingerprint (1024), rdkit_physchem (128)

### 3) Knowledge Graph (Drug Repurposing)
- **kg_edges.csv**: head_id, relation, tail_id, evidence_score, source_db, pmid, timestamp, citation
- **kg_nodes.csv**: node_id, node_type, name, synonyms, atc_codes, ontology_ids

### 4) Molecule Pool
- **molecule_pool_meta.csv**: standardized molecules with QED, Lipinski rule violations, splits
- **molecule_pool_vectors.parquet**: fingerprints + physchem

### 5) Protein Binding Sites
- **binding_sites_meta.csv**: pockets and centroids; optional protein vectors

### 6) Virtual Screening
- **virtual_screening_meta.csv**: docking_score, hit_label
- **virtual_screening_vectors.parquet**: molecule fingerprint (+ optional protein embedding)

## Pipeline Architecture

```
Data Sources → Connectors → Feature Generation → Standardization → Output
    ↓              ↓              ↓                ↓              ↓
  ChEMBL      chembl_connector   molecular_features   QC/Validation   CSV + Parquet
  UniProt     uniprot_connector  protein_features     Split Assignment
  PubChem     (future)           (extensible)         Provenance
```

## Components

### Connectors
- `connectors/chembl_connector.py`: ChEMBL REST API client
- `connectors/uniprot_connector.py`: UniProt REST API client

### Feature Generators
- `features/molecular_features.py`: RDKit-based molecular descriptors and fingerprints
- `features/protein_features.py`: Sequence-based protein embeddings and descriptors

### Build Scripts
- `scripts/build_datasets.py`: Main orchestration script with CLI interface

### CI/CD
- `.github/workflows/build-datasets.yml`: Automated dataset building and validation

## Configuration

Set environment variables for API rate limiting and caching:

```bash
export CHEMBL_CACHE_DIR=~/.cache/chembl
export UNIPROT_CACHE_DIR=~/.cache/uniprot
export PIPELINE_LOG_LEVEL=INFO
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this dataset pipeline in your research, please cite:

```bibtex
@software{drug_dataset_pipeline,
  title={AI-Augmented Drug Discovery Dataset Pipeline},
  author={Drug-Dataset Contributors},
  year={2024},
  url={https://github.com/KrithikVishal/Drug-Dataset}
}
```
