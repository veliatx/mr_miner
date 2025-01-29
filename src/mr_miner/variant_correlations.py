"""Variant correlation analysis tools."""
from pathlib import Path
import collections
import itertools
import json
import urllib
import urllib.request
import urllib.error
import time
import warnings
import logging
import io
from typing import Collection, Iterable

import pandas as pd

from mr_miner import constants
from mr_miner.constants import (
    DEFAULT_POP
)
from mr_miner.plink_wrapper import PlinkWrapper
from mr_miner.utilities import reflect_tri, replace_df_values

logger = logging.getLogger(__name__)


class VariantCorrelations:
    def __init__(self, 
                 spdi_rsid_translator,
                 plink_wrapper: PlinkWrapper,
                 correlations_cache_fpath: Path | str | None = None,
                 pop: str = DEFAULT_POP):
        """
        Initialize variant correlations handler
        
        Args:
            spdi_rsid_translator: Translator object for SPDI to rsID conversion
            plink_wrapper: PlinkWrapper instance for handling PLINK operations
            correlations_cache_fpath: Optional path to cache correlation results
            pop: Population code (e.g. 'EUR')
            verbose: Whether to print verbose output
        """
        self.spdi_rsid_translator = spdi_rsid_translator
        self.plink_wrapper = plink_wrapper
        self.correlations_cache_fpath = Path(correlations_cache_fpath) if correlations_cache_fpath else None
        self.pop = pop
        self._missed_rsids = set()
        
        self._correlations = {}
        if self.correlations_cache_fpath and self.correlations_cache_fpath.exists():
            self._load_from_cache()

    def _log_print(self, msg):
        if self.verbose:
            print(msg)

    def _load_from_file(self):
        try:
            with open(self.correlations_cache_fpath, 'rt') as data_file:
                for line in data_file:
                    splat = line.strip().split('\t')
                    if len(splat) == 1:
                        self._missed_rsids.add(splat[0])
                    else:
                        rsid1, rsid2, corr = splat
                        corr = float(corr)
    
                        if rsid1 not in self._correlations:
                            self._correlations[rsid1] = {rsid2: corr}
                        else:
                            self._correlations[rsid1][rsid2] = corr

        except FileNotFoundError:
            pass

    def _save_to_file(self, fpath: Path | str):
        with open(fpath, 'wt') as data_file:
            for rsid1 in self._correlations:
                for rsid2 in self._correlations[rsid1]:
                    data_file.write(f'{rsid1}\t{rsid2}\t{self._correlations[rsid1][rsid2]}\n')
            for rsid in sorted(self._missed_rsids):
                data_file.write(rsid + '\n')

    def _rewrite_cache(self):
        self._save_to_file(self.correlations_cache_fpath)

    def get_matrix(self, spdis: Iterable[str], method: str='local',
                  delete_temp_files: bool=True):
        spdi_list = sorted(set(spdis))
        logger.debug(f'Translating {len(spdi_list)} SPDIs')

        rsid_to_spdi = {}
        rsid_list = self.spdi_rsid_translator.translate_spdis_to_rsids(spdi_list, as_set=False)
        for spdi, rsid in zip(spdi_list, rsid_list):
            rsid_to_spdi[rsid] = spdi
        rsids = set(rsid_to_spdi.keys())
        logger.debug(f'Geting correlations for {len(rsids)} RSIDs')

        corr_matrix = self.get_matrix_rsids(rsids=rsids, method=method, delete_temp_files=delete_temp_files)
        back_translated_spdis = [rsid_to_spdi[rsid] for rsid in corr_matrix.index]
        
        corr_matrix.index = back_translated_spdis
        corr_matrix.columns = back_translated_spdis
        if '' in corr_matrix.index:
            corr_matrix = corr_matrix.drop('', axis=0).drop('', axis=1)

        return corr_matrix
    
    def get_matrix_rsids(self, rsids: Iterable[str], method: str='local',
                  delete_temp_files: bool=True):
      
        rsid_list = sorted(set(rsids))
        logger.debug(f'Geting correlations for {", ".join(rsid_list)}')

        matrix_data = collections.defaultdict(lambda: {})
        
        query_rsid_queue = set([])
        missing_pairs = collections.defaultdict(lambda: set([]))
        set([])
        new_triplets = set([])
                
        for rsid1, rsid2 in itertools.combinations(rsid_list, 2):
            if rsid1 not in self._missed_rsids and rsid2 not in self._missed_rsids:
                # Correlations are 0 for variants on different chromosomes
                # if rsid1.split(':')[0] != rsid2.split(':')[0]:
                #     matrix_data[rsid1][rsid2] = 0.0
                # # If we already have data on this pair, copy them into the result
                if rsid1 in self._correlations and rsid2 in self._correlations[rsid1]:
                    matrix_data[rsid1][rsid2] = self._correlations[rsid1][rsid2]
                # Otherwise we need to query an external resource for this pair
                else:
                    query_rsid_queue.update([rsid1, rsid2])
                    missing_pairs[rsid1].add(rsid2)

        if len(query_rsid_queue) > 0:
            logger.debug(f'Need to query for {", ".join(query_rsid_queue)}')
                
            query_result_matrix = self.query_ld_resource(query_rsid_queue, method=method, delete_temp_files=delete_temp_files)

            logger.debug(f'Got response: {query_result_matrix}')
            found_rsids = set(query_result_matrix.index)
            self._update_missing_rsids(query_rsid_queue.difference(found_rsids)) 
    
            query_result_dict = query_result_matrix.to_dict()
            
            for rsid1, rsid_set in missing_pairs.items():
                if rsid1 not in found_rsids:
                    break
                for rsid2 in rsid_set:
                    if rsid2 in found_rsids:
                        corr = query_result_dict[rsid1][rsid2]
                        new_triplets.add((rsid1, rsid2, corr))
                        matrix_data[rsid1][rsid2] = corr
    
            self._update_correlations(new_triplets)
      
        # # Self-correlations always 1.0
        for rsid in rsid_list:
            if rsid not in self._missed_rsids:
                matrix_data[rsid][rsid] = 1.0

        # print(matrix_data)

        corr_matrix = pd.DataFrame(matrix_data).sort_index(axis=0).sort_index(axis=1).fillna(0)       
        # print(corr_matrix)
        corr_matrix = replace_df_values(corr_matrix, reflect_tri(corr_matrix.values))
                    
        return corr_matrix.astype(float)

    def _update_missing_rsids(self, new_rsids):
        if self.correlations_cache_fpath:
            with open(self.correlations_cache_fpath, 'at') as data_file:
                for rsid in new_rsids.difference(self._missed_rsids):
                    data_file.write(rsid + '\n')
        self._missed_rsids.update(new_rsids)

    def _update_correlations(self, new_triplets):
        if self.correlations_cache_fpath:
            with open(self.correlations_cache_fpath, 'at') as data_file:
                for rsid1, rsid2, corr in new_triplets:
                    if rsid1 in self._correlations:
                        if rsid2 in self._correlations[rsid1]:
                            continue     
                        else:
                            self._correlations[rsid1][rsid2] = corr
                    else: 
                        self._correlations[rsid1] = {rsid2: corr}
                    
                    data_file.write('\t'.join((rsid1, rsid2, str(corr))) + '\n')

    def query_ld_resource(self, rsids: Collection[str], method='local', delete_temp_files: bool=True):
        # spdi_to_snp_id, snp_id_to_spdi = self.prepare_ld_matrix_query_snps(spdis)
        
        # snp_ids = []
        # for snp_list in spdi_to_snp_id.values():
        #     snp_ids += snp_list

        if method == 'local':
            query_results = self.execute_ldmatrix_query_local(rsids, delete_temp_files=delete_temp_files)
        else:
            query_results = self.execute_ldmatrix_query_post(rsids)
 
        return query_results
                            
    def prepare_ld_matrix_query_snps(self, spdis, drop_rsid_misses=True, pos_offset: int=1):
        spdi_to_snp_id = collections.defaultdict(lambda: [])
        snp_id_to_spdi = collections.defaultdict(lambda: [])
    
        for spdi in spdis:
            rsids = self.spdi_map[spdi]
        
            if rsids is not None:
                spdi_to_snp_id[spdi] = rsids
                for rsid in rsids:
                    snp_id_to_spdi[rsid].append(spdi)
                    
            elif not drop_rsid_misses:
                chrom, pos, ref, alt = spdi.split(':')
                loc_id = f'{chrom}:{int(pos) + pos_offset}'
                spdi_to_snp_id[spdi].append(loc_id)
                snp_id_to_spdi[loc_id].append(spdi)
        
        return spdi_to_snp_id, snp_id_to_spdi

    def execute_ldmatrix_query_local(self, snp_ids: Iterable[str], delete_temp_files: bool=True) -> pd.DataFrame:
        return self.plink_wrapper.get_corr_matrix(snp_ids, delete_temp_files=delete_temp_files)
    
    def execute_ldmatrix_query_get(self, snp_ids: Iterable[str]) -> pd.DataFrame:
        if len(snp_ids) > constants.DEFAULT_MAX_GET_VARIANTS:
            warnings.warn(f'Max {constants.DEFAULT_MAX_GET_VARIANTS} variants can be queried with a GET request, got {len(snp_ids)}.')
            
        URL_TEMPLATE = 'https://ldlink.nih.gov/LDlinkRest/ldmatrix?snps={snp_list}&pop={pop}&r2_d=d&genome_build={genome_build}&token={token}'
        
        snp_list = '%0A'.join(snp_ids)
        url = URL_TEMPLATE.format(snp_list=snp_list, pop=self.pop, genome_build=self.genome_build, token=constants.LDLINK_API_TOKEN)
        resp = make_request_with_retries(url)
        
        return pd.read_csv(io.StringIO(resp), sep='\t', index_col=0)
    
    def execute_ldmatrix_query_post(self, snp_ids: Collection[str]) -> pd.DataFrame:
        if len(snp_ids) > constants.DEFAULT_MAX_POST_VARIANTS:
            warnings.warn(f'Max {constants.DEFAULT_MAX_POST_VARIANTS} variants can be queried with a POST request, got {len(snp_ids)}.')
        
        URL_TEMPLATE = 'https://ldlink.nih.gov/LDlinkRest/ldmatrix?token={token}'
        url = URL_TEMPLATE.format(token=self.ldlink_api_token)
    
        data_dict = {"snps": '\n'.join(snp_ids),
                     "pop": self.pop, 
                     "r2_d": "d",
                     "genome_build": self.genome_build}
        data = json.dumps(data_dict).encode('utf-8')
        
        req = urllib.request.Request(url=url,
                                     data=data,
                                     headers={"Content-Type": "application/json"},
                                     method='POST')
        resp = make_request_with_retries(req)

        if 'error' in resp:
            raise urllib.error.URLError(resp)
        
        return pd.read_csv(io.StringIO(resp), sep='\t', index_col=0).astype(float)



def make_request_with_retries(req, retries=constants.DEFAULT_RETRIES, timeout=constants.REQUEST_TIMEOUT, backoff_factor=constants.DEFAULT_BACKOFF_FACTOR):
    attempt = 0
    while attempt < retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_text = response.read().decode()
                if 'error' in response_text:
                    raise urllib.error.URLError(response_text)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            attempt += 1
            timeout *= backoff_factor
            print(f"Attempt {attempt} failed with error: {e}. Retrying in {timeout} seconds...")
            time.sleep(timeout)
        else:
            return response_text

    raise TimeoutError(f"Failed to retrieve the URL after {retries} attempts")

