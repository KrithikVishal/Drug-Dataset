"""
Protein feature generation for embeddings and descriptors.
"""

import logging
import numpy as np
from typing import List, Optional
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import hashlib


class ProteinFeatureGenerator:
    """Generate protein embeddings and descriptors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_embeddings(self, sequences: List[str], 
                          method: str = "one_hot") -> np.ndarray:
        """Generate protein sequence embeddings."""
        self.logger.info(f"Generating {method} embeddings for {len(sequences)} sequences...")
        
        if method == "one_hot":
            return self._generate_one_hot_embeddings(sequences)
        elif method == "physicochemical":
            return self._generate_physicochemical_features(sequences)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    def _generate_one_hot_embeddings(self, sequences: List[str], 
                                   max_length: int = 1000) -> np.ndarray:
        """Generate one-hot encoded protein sequences."""
        aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20  # X for unknown
        }
        
        embeddings = []
        
        for seq in sequences:
            try:
                # Truncate or pad sequence
                seq = seq[:max_length] if len(seq) > max_length else seq
                
                # One-hot encode
                encoding = np.zeros((max_length, 21), dtype=np.float32)
                for i, aa in enumerate(seq):
                    idx = aa_to_idx.get(aa.upper(), 20)  # Use 'X' for unknown
                    encoding[i, idx] = 1.0
                
                # Flatten to 1D vector
                embeddings.append(encoding.flatten())
                
            except Exception as e:
                self.logger.warning(f"Failed to encode sequence: {e}")
                embeddings.append(np.zeros(max_length * 21, dtype=np.float32))
        
        embeddings_array = np.array(embeddings)
        self.logger.info(f"Generated one-hot embeddings with shape {embeddings_array.shape}")
        return embeddings_array
    
    def _generate_physicochemical_features(self, sequences: List[str]) -> np.ndarray:
        """Generate physicochemical protein features."""
        features = []
        
        for seq in sequences:
            try:
                if not seq or len(seq) < 10:  # Skip very short sequences
                    features.append(np.zeros(15, dtype=np.float32))
                    continue
                
                # Use BioPython for analysis
                analysis = ProteinAnalysis(seq)
                
                feature_vector = []
                
                # Basic properties
                feature_vector.append(len(seq))  # Length
                feature_vector.append(analysis.molecular_weight())  # MW
                feature_vector.append(analysis.aromaticity())  # Aromaticity
                feature_vector.append(analysis.instability_index())  # Instability
                feature_vector.append(analysis.isoelectric_point())  # pI
                
                # Amino acid composition (20 features -> reduced to 5 groups)
                aa_composition = analysis.get_amino_acids_percent()
                
                # Group amino acids by properties
                hydrophobic = sum(aa_composition.get(aa, 0) for aa in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'])
                hydrophilic = sum(aa_composition.get(aa, 0) for aa in ['S', 'T', 'N', 'Q'])
                positive = sum(aa_composition.get(aa, 0) for aa in ['R', 'K', 'H'])
                negative = sum(aa_composition.get(aa, 0) for aa in ['D', 'E'])
                aromatic = sum(aa_composition.get(aa, 0) for aa in ['F', 'W', 'Y'])
                
                feature_vector.extend([hydrophobic, hydrophilic, positive, negative, aromatic])
                
                # Secondary structure propensity (simplified)
                helix_propensity = sum(aa_composition.get(aa, 0) for aa in ['A', 'E', 'L'])
                sheet_propensity = sum(aa_composition.get(aa, 0) for aa in ['V', 'I', 'F'])
                turn_propensity = sum(aa_composition.get(aa, 0) for aa in ['G', 'P', 'S'])
                
                feature_vector.extend([helix_propensity, sheet_propensity, turn_propensity])
                
                # Handle NaN values
                feature_vector = [0.0 if np.isnan(x) or np.isinf(x) else float(x) 
                                for x in feature_vector]
                
                features.append(np.array(feature_vector, dtype=np.float32))
                
            except Exception as e:
                self.logger.warning(f"Failed to generate features for sequence: {e}")
                features.append(np.zeros(15, dtype=np.float32))
        
        features_array = np.array(features)
        self.logger.info(f"Generated physicochemical features with shape {features_array.shape}")
        return features_array
    
    def generate_sequence_descriptors(self, sequences: List[str]) -> np.ndarray:
        """Generate sequence-based descriptors."""
        descriptors = []
        
        for seq in sequences:
            try:
                if not seq:
                    descriptors.append(np.zeros(10, dtype=np.float32))
                    continue
                
                # Basic sequence properties
                length = len(seq)
                
                # Amino acid counts
                aa_counts = {}
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    aa_counts[aa] = seq.count(aa) / length if length > 0 else 0
                
                # Compute various features
                features = [
                    length,
                    seq.count('C') / length if length > 0 else 0,  # Cysteine fraction
                    seq.count('P') / length if length > 0 else 0,  # Proline fraction
                    (seq.count('R') + seq.count('K')) / length if length > 0 else 0,  # Basic residues
                    (seq.count('D') + seq.count('E')) / length if length > 0 else 0,  # Acidic residues
                    (seq.count('F') + seq.count('W') + seq.count('Y')) / length if length > 0 else 0,  # Aromatic
                    (seq.count('A') + seq.count('V') + seq.count('I') + seq.count('L') + seq.count('M')) / length if length > 0 else 0,  # Hydrophobic
                    seq.count('G') / length if length > 0 else 0,  # Glycine fraction
                    len(set(seq)) / 20.0,  # Amino acid diversity
                    self._calculate_charge(seq)  # Net charge
                ]
                
                descriptors.append(np.array(features, dtype=np.float32))
                
            except Exception as e:
                self.logger.warning(f"Failed to generate descriptors for sequence: {e}")
                descriptors.append(np.zeros(10, dtype=np.float32))
        
        descriptors_array = np.array(descriptors)
        self.logger.info(f"Generated sequence descriptors with shape {descriptors_array.shape}")
        return descriptors_array
    
    def _calculate_charge(self, sequence: str, ph: float = 7.0) -> float:
        """Calculate net charge of protein at given pH."""
        try:
            analysis = ProteinAnalysis(sequence)
            return analysis.charge_at_pH(ph)
        except:
            return 0.0
    
    def generate_binding_site_features(self, sequences: List[str], 
                                     binding_sites: List[List[int]]) -> np.ndarray:
        """Generate features for protein binding sites."""
        features = []
        
        for seq, sites in zip(sequences, binding_sites):
            try:
                site_features = []
                
                for site_residues in sites:
                    if not site_residues or min(site_residues) < 0 or max(site_residues) >= len(seq):
                        site_features.extend([0.0] * 5)  # Default features
                        continue
                    
                    # Extract binding site sequence
                    site_seq = ''.join([seq[i] for i in site_residues])
                    
                    if not site_seq:
                        site_features.extend([0.0] * 5)
                        continue
                    
                    # Calculate site-specific features
                    site_length = len(site_seq)
                    hydrophobic_ratio = sum(1 for aa in site_seq if aa in 'AVILMFYW') / site_length
                    charged_ratio = sum(1 for aa in site_seq if aa in 'RKDE') / site_length
                    aromatic_ratio = sum(1 for aa in site_seq if aa in 'FWY') / site_length
                    polar_ratio = sum(1 for aa in site_seq if aa in 'STNQ') / site_length
                    
                    site_features.extend([
                        site_length,
                        hydrophobic_ratio,
                        charged_ratio,
                        aromatic_ratio,
                        polar_ratio
                    ])
                
                # Pad or truncate to fixed size (e.g., 3 sites * 5 features = 15)
                site_features = site_features[:15]
                while len(site_features) < 15:
                    site_features.append(0.0)
                
                features.append(np.array(site_features, dtype=np.float32))
                
            except Exception as e:
                self.logger.warning(f"Failed to generate binding site features: {e}")
                features.append(np.zeros(15, dtype=np.float32))
        
        features_array = np.array(features)
        self.logger.info(f"Generated binding site features with shape {features_array.shape}")
        return features_array