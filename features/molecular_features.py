"""
Molecular feature generation using RDKit for fingerprints and descriptors.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors
from sklearn.preprocessing import StandardScaler


class MolecularFeatureGenerator:
    """Generate molecular fingerprints and descriptors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def standardize_molecules(self, smiles_list: List[str]) -> pd.DataFrame:
        """Standardize SMILES strings."""
        self.logger.info(f"Standardizing {len(smiles_list)} molecules...")
        
        records = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Neutralize charges
                neutralized_mol = self._neutralize_mol(mol)
                
                # Remove salts
                desalted_mol = self._remove_salts(neutralized_mol)
                
                # Standardize tautomers (simplified)
                standardized_smiles = Chem.MolToSmiles(desalted_mol, canonical=True)
                
                record = {
                    'original_smiles': smiles,
                    'canonical_smiles': standardized_smiles,
                    'inchi_key': Chem.MolToInchiKey(desalted_mol),
                    'is_neutralized': True,
                    'salt_removed': True,
                    'standardized': True,
                    'stereo_flag': 'defined' if '@' in standardized_smiles else 'undefined'
                }
                
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Failed to standardize {smiles}: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Standardized {len(df)} molecules")
        return df
    
    def _neutralize_mol(self, mol):
        """Neutralize charged molecules."""
        # Simplified neutralization
        pattern = Chem.MolFromSmarts("[+1!H0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        
        return mol
    
    def _remove_salts(self, mol):
        """Remove salt components."""
        return Chem.rdMolStandardize.rdMolStandardize.ChargeParent(mol)
    
    def generate_fingerprints(self, smiles_list: List[str], 
                            fp_type: str = "morgan", radius: int = 2, 
                            n_bits: int = 1024) -> np.ndarray:
        """Generate molecular fingerprints."""
        self.logger.info(f"Generating {fp_type} fingerprints for {len(smiles_list)} molecules...")
        
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    fingerprints.append(np.zeros(n_bits, dtype=np.int8))
                    continue
                
                if fp_type == "morgan":
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                elif fp_type == "maccs":
                    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                    n_bits = 167  # MACCS keys are 167 bits
                else:
                    raise ValueError(f"Unknown fingerprint type: {fp_type}")
                
                fp_array = np.array(fp, dtype=np.int8)
                fingerprints.append(fp_array)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate fingerprint for {smiles}: {e}")
                fingerprints.append(np.zeros(n_bits, dtype=np.int8))
        
        fingerprints_array = np.array(fingerprints)
        self.logger.info(f"Generated fingerprints with shape {fingerprints_array.shape}")
        return fingerprints_array
    
    def generate_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """Generate RDKit molecular descriptors."""
        self.logger.info(f"Generating descriptors for {len(smiles_list)} molecules...")
        
        descriptor_functions = [
            Descriptors.MolWt, Descriptors.MolLogP, Descriptors.NumHDonors,
            Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds,
            Descriptors.TPSA, Descriptors.NumAliphaticCarbocycles,
            Descriptors.NumAliphaticHeterocycles, Descriptors.NumAromaticCarbocycles,
            Descriptors.NumAromaticHeterocycles, Descriptors.RingCount,
            Descriptors.FractionCsp3, Descriptors.NumHeteroatoms,
            Descriptors.NumSaturatedCarbocycles, Descriptors.NumSaturatedHeterocycles,
            Descriptors.BalabanJ, Descriptors.BertzCT, Descriptors.Chi0, Descriptors.Chi1,
            Descriptors.HallKierAlpha, Descriptors.Ipc, Descriptors.Kappa1, Descriptors.Kappa2,
            Descriptors.Kappa3, Descriptors.LabuteASA, Descriptors.PEOE_VSA1,
            Descriptors.PEOE_VSA2, Descriptors.PEOE_VSA3, Descriptors.PEOE_VSA4,
            Descriptors.PEOE_VSA5, Descriptors.PEOE_VSA6, Descriptors.SMR_VSA1,
            Descriptors.SMR_VSA10, Descriptors.SMR_VSA2, Descriptors.SMR_VSA3,
            Descriptors.SMR_VSA4, Descriptors.SMR_VSA5, Descriptors.SMR_VSA6,
            Descriptors.SMR_VSA7, Descriptors.SMR_VSA8, Descriptors.SMR_VSA9,
            Descriptors.SlogP_VSA1, Descriptors.SlogP_VSA10, Descriptors.SlogP_VSA11,
            Descriptors.SlogP_VSA12, Descriptors.SlogP_VSA2, Descriptors.SlogP_VSA3,
            Descriptors.SlogP_VSA4, Descriptors.SlogP_VSA5, Descriptors.SlogP_VSA6,
            Descriptors.SlogP_VSA7, Descriptors.SlogP_VSA8, Descriptors.SlogP_VSA9,
            Descriptors.VSA_EState1, Descriptors.VSA_EState10, Descriptors.VSA_EState2,
            Descriptors.VSA_EState3, Descriptors.VSA_EState4, Descriptors.VSA_EState5,
            Descriptors.VSA_EState6, Descriptors.VSA_EState7, Descriptors.VSA_EState8,
            Descriptors.VSA_EState9, Descriptors.EState_VSA1, Descriptors.EState_VSA10,
            Descriptors.EState_VSA11, Descriptors.EState_VSA2, Descriptors.EState_VSA3,
            Descriptors.EState_VSA4, Descriptors.EState_VSA5, Descriptors.EState_VSA6,
            Descriptors.EState_VSA7, Descriptors.EState_VSA8, Descriptors.EState_VSA9
        ]
        
        descriptors = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    descriptors.append(np.zeros(len(descriptor_functions), dtype=np.float32))
                    continue
                
                desc_values = []
                for desc_fn in descriptor_functions:
                    try:
                        value = desc_fn(mol)
                        # Handle infinity and NaN values
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        desc_values.append(float(value))
                    except:
                        desc_values.append(0.0)
                
                descriptors.append(np.array(desc_values, dtype=np.float32))
                
            except Exception as e:
                self.logger.warning(f"Failed to generate descriptors for {smiles}: {e}")
                descriptors.append(np.zeros(len(descriptor_functions), dtype=np.float32))
        
        descriptors_array = np.array(descriptors)
        
        # Standardize descriptors
        if len(descriptors_array) > 1:
            descriptors_array = self.scaler.fit_transform(descriptors_array).astype(np.float32)
        
        self.logger.info(f"Generated descriptors with shape {descriptors_array.shape}")
        return descriptors_array
    
    def calculate_qed(self, smiles_list: List[str]) -> pd.DataFrame:
        """Calculate QED (Quantitative Estimate of Drug-likeness) scores."""
        qed_scores = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    qed_scores.append(0.0)
                else:
                    qed_scores.append(qed(mol))
            except:
                qed_scores.append(0.0)
        
        return pd.DataFrame({'qed_score': qed_scores})
    
    def calculate_lipinski_violations(self, smiles_list: List[str]) -> pd.DataFrame:
        """Calculate Lipinski rule of five violations."""
        violations = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    violations.append(4)  # Max violations
                    continue
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = NumHDonors(mol)
                hba = NumHAcceptors(mol)
                
                violation_count = 0
                if mw > 500:
                    violation_count += 1
                if logp > 5:
                    violation_count += 1
                if hbd > 5:
                    violation_count += 1
                if hba > 10:
                    violation_count += 1
                
                violations.append(violation_count)
                
            except:
                violations.append(4)
        
        return pd.DataFrame({'lipinski_violations': violations})