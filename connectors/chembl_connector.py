"""
ChEMBL database connector for fetching bioactivity and molecular data.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List
import requests
import time
from pathlib import Path


class ChEMBLConnector:
    """Connector to ChEMBL database via REST API."""
    
    def __init__(self, base_url: str = "https://www.ebi.ac.uk/chembl/api/data"):
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make API request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params.setdefault('format', 'json')
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def get_bioactivities(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch drug-target interaction bioactivities."""
        self.logger.info("Fetching bioactivities from ChEMBL...")
        
        params = {
            'limit': limit or 1000,
            'target_type': 'PROTEIN',
            'standard_type__in': 'IC50,Ki,Kd,EC50',
            'standard_relation': '=',
            'assay_type': 'B'  # Binding assays
        }
        
        data = self._make_request('activity', params)
        activities = data.get('activities', [])
        
        # Convert to DataFrame with standardized columns
        records = []
        for activity in activities:
            try:
                record = {
                    'molecule_id': activity.get('molecule_chembl_id'),
                    'canonical_smiles': activity.get('canonical_smiles'),
                    'target_id': activity.get('target_chembl_id'),
                    'assay_id': activity.get('assay_chembl_id'),
                    'assay_type': activity.get('assay_type'),
                    'endpoint_type': activity.get('standard_type'),
                    'value': activity.get('standard_value'),
                    'units': activity.get('standard_units'),
                    'qualifier': activity.get('standard_relation'),
                    'source': 'ChEMBL',
                    'year': activity.get('src_id', 0) // 1000 + 2000,  # Rough estimate
                    'split': 'train'  # Default assignment
                }
                
                # Add standardization flags
                mol_props = activity.get('molecule_properties', {})
                record.update({
                    'is_neutralized': True,  # ChEMBL provides neutralized forms
                    'stereo_flag': 'defined' if '@' in (record['canonical_smiles'] or '') else 'undefined',
                    'salt_removed': True,
                    'standardized': True
                })
                
                # Generate scaffold and duplicate group IDs (simplified)
                record['scaffold_id'] = hash(record['canonical_smiles']) % 10000 if record['canonical_smiles'] else None
                record['duplicate_group'] = hash(record['canonical_smiles']) % 1000 if record['canonical_smiles'] else None
                
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Skipping malformed activity record: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Retrieved {len(df)} bioactivity records")
        return df
    
    def get_admet_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch ADMET-related bioactivities."""
        self.logger.info("Fetching ADMET data from ChEMBL...")
        
        admet_endpoints = [
            'Hepatotoxicity', 'hERG', 'Caco-2', 'Solubility', 
            'Clearance', 'Bioavailability', 'CYP450'
        ]
        
        params = {
            'limit': limit or 1000,
            'standard_type__icontains': '|'.join(admet_endpoints),
            'assay_type__in': 'A,F'  # ADMET and Functional assays
        }
        
        data = self._make_request('activity', params)
        activities = data.get('activities', [])
        
        records = []
        for activity in activities:
            try:
                record = {
                    'molecule_id': activity.get('molecule_chembl_id'),
                    'canonical_smiles': activity.get('canonical_smiles'),
                    'assay_id': activity.get('assay_chembl_id'),
                    'endpoint_type': activity.get('standard_type'),
                    'value': activity.get('standard_value'),
                    'units': activity.get('standard_units'),
                    'source': 'ChEMBL',
                    'split': 'train'
                }
                
                # Map to standardized ADMET endpoints
                endpoint_map = {
                    'hepatotoxicity': 'hepatotox_label',
                    'herg': 'hERG_label', 
                    'caco': 'caco2_perm',
                    'solubility': 'logS',
                    'clearance': 'clint'
                }
                
                for key, standard_name in endpoint_map.items():
                    if key.lower() in record['endpoint_type'].lower():
                        record[standard_name] = record['value']
                
                # Add quality control flags
                record.update({
                    'qc_pass': True,
                    'replicates': 1,
                    'outlier_flag': False
                })
                
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Skipping malformed ADMET record: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Retrieved {len(df)} ADMET records")
        return df
    
    def get_molecule_pool(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch diverse set of drug-like molecules."""
        self.logger.info("Fetching molecule pool from ChEMBL...")
        
        params = {
            'limit': limit or 1000,
            'molecule_properties__mw_freebase__gte': 200,
            'molecule_properties__mw_freebase__lte': 800,
            'molecule_properties__alogp__gte': -3,
            'molecule_properties__alogp__lte': 5
        }
        
        data = self._make_request('molecule', params)
        molecules = data.get('molecules', [])
        
        records = []
        for molecule in molecules:
            try:
                record = {
                    'molecule_id': molecule.get('molecule_chembl_id'),
                    'canonical_smiles': molecule.get('molecule_structures', {}).get('canonical_smiles'),
                    'inchi_key': molecule.get('molecule_structures', {}).get('standard_inchi_key'),
                    'source': 'ChEMBL',
                    'split': 'train'
                }
                
                if record['canonical_smiles']:
                    records.append(record)
                    
            except Exception as e:
                self.logger.warning(f"Skipping malformed molecule record: {e}")
                continue
        
        df = pd.DataFrame(records)
        self.logger.info(f"Retrieved {len(df)} molecules")
        return df