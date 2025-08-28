# Processed Datasets (Real-time friendly schemas)

Each task emits a thin CSV + a vector Parquet:
- CSV: compact metadata, IDs, quality flags, provenance, `scaffold_id`, `split`.
- Parquet: array columns for vectors (`mol_fingerprint`, `rdkit_physchem`, optional `protein_embedding`).

Schemas

1) DTI
- dti_meta.csv
  - molecule_id, canonical_smiles, inchi_key
  - is_neutralized, stereo_flag, salt_removed, standardized
  - scaffold_id, duplicate_group
  - target_id, uniprot_id, organism, target_class
  - assay_id, assay_type, endpoint_type (Ki/Kd/IC50/EC50)
  - value (nM), qualifier (>, <, =), units (nM)
  - pH, temperature, assay_confidence
  - source, citation, year, split
- dti_vectors.parquet
  - molecule_id, target_id
  - mol_fingerprint (int8[1024])
  - rdkit_physchem (float32[128])
  - protein_embedding (float32[N] or null)

2) ADMET
- admet_meta.csv
  - molecule_id, canonical_smiles, inchi_key
  - standardization flags
  - endpoints (e.g., hepatotox_label, hERG_label, caco2_perm, logS, clint)
  - assay meta: assay_id, protocol, units, batch_id, lab
  - quality: qc_pass, replicates, stdev, outlier_flag
  - source, citation, year, split, scaffold_id, duplicate_group
- admet_vectors.parquet
  - molecule_id, mol_fingerprint (1024), rdkit_physchem (128)

3) KG (repurposing)
- kg_edges.csv: head_id, relation, tail_id, evidence_score, source_db, pmid, timestamp, citation
- kg_nodes.csv: node_id, node_type, name, synonyms, atc_codes, ontology_ids

4) Molecule Pool
- molecule_pool_meta.csv: standardized molecules with QED, Lipinski rule violations, splits
- molecule_pool_vectors.parquet: fingerprints + physchem

5) Protein Binding Sites
- binding_sites_meta.csv: pockets and centroids; optional protein vectors

6) Virtual Screening
- virtual_screening_meta.csv: docking_score, hit_label
- virtual_screening_vectors.parquet: molecule fingerprint (+ optional protein embedding)

Build
```bash
python scripts/build_datasets.py --all --chembl --uniprot --limit-chembl 200000 --out data/
```
