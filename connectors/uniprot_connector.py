"""
UniProt database connector for fetching protein sequence and annotation data.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List
import requests
import time
import xml.etree.ElementTree as ET


class UniProtConnector:
    """Connector to UniProt database via REST API."""
    
    def __init__(self, base_url: str = "https://rest.uniprot.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> requests.Response:
        """Make API request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def get_target_info(self, target_ids: List[str]) -> pd.DataFrame:
        """Fetch protein target information from UniProt."""
        self.logger.info(f"Fetching target info for {len(target_ids)} targets...")
        
        records = []
        
        for target_id in target_ids:
            try:
                # Convert ChEMBL target ID to UniProt accession if needed
                uniprot_id = self._map_chembl_to_uniprot(target_id)
                if not uniprot_id:
                    continue
                
                # Fetch protein entry
                params = {
                    'accession': uniprot_id,
                    'format': 'json'
                }
                
                response = self._make_request(f"uniprotkb/{uniprot_id}", params)
                data = response.json()
                
                record = {
                    'target_id': target_id,
                    'uniprot_id': uniprot_id,
                    'organism': self._extract_organism(data),
                    'target_class': self._extract_target_class(data),
                    'sequence': self._extract_sequence(data),
                    'gene_name': self._extract_gene_name(data),
                    'protein_name': self._extract_protein_name(data)
                }
                
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch target {target_id}: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Retrieved {len(df)} target records")
        return df
    
    def _map_chembl_to_uniprot(self, chembl_id: str) -> Optional[str]:
        """Map ChEMBL target ID to UniProt accession."""
        # Simplified mapping - in practice would use ChEMBL API
        if chembl_id and chembl_id.startswith('CHEMBL'):
            # Mock mapping for demonstration
            mapping = {
                'CHEMBL279': 'P35367',  # Example: Histamine H1 receptor
                'CHEMBL204': 'P08913',  # Example: Alpha-2A adrenergic receptor
                'CHEMBL252': 'P41143',  # Example: Delta-type opioid receptor
            }
            return mapping.get(chembl_id)
        return None
    
    def _extract_organism(self, data: Dict) -> str:
        """Extract organism information."""
        try:
            organism = data.get('organism', {})
            return organism.get('scientificName', 'Unknown')
        except:
            return 'Unknown'
    
    def _extract_target_class(self, data: Dict) -> str:
        """Extract target classification."""
        try:
            # Look for GO terms, protein families, etc.
            annotations = data.get('dbReferences', [])
            for ref in annotations:
                if ref.get('type') == 'InterPro':
                    return ref.get('properties', {}).get('entry name', 'GPCR')
            return 'Enzyme'  # Default
        except:
            return 'Unknown'
    
    def _extract_sequence(self, data: Dict) -> str:
        """Extract protein sequence."""
        try:
            return data.get('sequence', {}).get('value', '')
        except:
            return ''
    
    def _extract_gene_name(self, data: Dict) -> str:
        """Extract gene name."""
        try:
            genes = data.get('genes', [])
            if genes:
                return genes[0].get('geneName', {}).get('value', '')
            return ''
        except:
            return ''
    
    def _extract_protein_name(self, data: Dict) -> str:
        """Extract protein name."""
        try:
            protein_desc = data.get('proteinDescription', {})
            rec_name = protein_desc.get('recommendedName', {})
            return rec_name.get('fullName', {}).get('value', '')
        except:
            return ''
    
    def get_binding_sites(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Fetch protein binding site information."""
        self.logger.info(f"Fetching binding sites for {len(uniprot_ids)} proteins...")
        
        records = []
        
        for uniprot_id in uniprot_ids:
            try:
                params = {
                    'accession': uniprot_id,
                    'format': 'json'
                }
                
                response = self._make_request(f"uniprotkb/{uniprot_id}", params)
                data = response.json()
                
                # Extract binding sites from features
                features = data.get('features', [])
                for feature in features:
                    if feature.get('type') in ['BINDING', 'ACT_SITE', 'SITE']:
                        record = {
                            'uniprot_id': uniprot_id,
                            'site_type': feature.get('type'),
                            'start_pos': feature.get('location', {}).get('start', {}).get('value'),
                            'end_pos': feature.get('location', {}).get('end', {}).get('value'),
                            'description': feature.get('description', ''),
                            'evidence': feature.get('evidences', [{}])[0].get('code', '')
                        }
                        records.append(record)
                        
            except Exception as e:
                self.logger.warning(f"Failed to fetch binding sites for {uniprot_id}: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Retrieved {len(df)} binding site records")
        return df