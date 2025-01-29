"""Variant ID handling utilities."""
import bz2
import csv
import gzip
import io
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd
from tqdm import tqdm

from mr_miner.chrom_mapper import ChromMapper
from mr_miner.config import MRConfig
from mr_miner.constants import (
    DBSNP_CHROM_NAMESPACE,
    DBSNP_SPDI_POS_OFFSET,
    DEFAULT_THREADS,
    THOUSAND_GENOMES_POS_OFFSET,
    WORKING_CHROM_NAMESPACE,
)
from mr_miner.plink_wrapper import PlinkWrapper
from mr_miner.utilities import iterate_cols, right_shift_variant_for_vep

logger = logging.getLogger(__name__)


class SpdiRsidTranslator(ABC):
    """
    Base class for SPDI to rsID translation.
    Provides interface and common functionality for translating between SPDI and rsID formats.
    """
    def __init__(self, cache_path: Optional[Union[Path, str]] = None):
        """
        Initialize translator with optional cache files.

        Args:
            cache_path: Path to directory containing cache files for storing translations
        """
        self._spdi_to_rsid: Dict[str, Set[str]] = {}
        self._rsid_to_spdi: Dict[str, Set[str]] = {}
        self._missing_spdis: Set[str] = set()
        self._missing_rsids: Set[str] = set()
        
        if cache_path:
            cache_path = Path(cache_path)
            self.spdi_to_rsid_cache_fpath = cache_path.joinpath('spdi_to_rsid_cache.txt.gz')
            self.rsid_to_spdi_cache_fpath = cache_path.joinpath('rsid_to_spdi_cache.txt.gz')
            os.makedirs(cache_path, exist_ok=True)
            self._load_cache()
        else:
            self.spdi_to_rsid_cache_fpath = None
            self.rsid_to_spdi_cache_fpath = None
  
    def _load_cache(self) -> None:
        """Load cached translations from both cache files."""
        # Load SPDI to rsID cache
        if self.spdi_to_rsid_cache_fpath and self.spdi_to_rsid_cache_fpath.exists():
            logger.info(f'Loading cached spdi-to-rsid translations from {self.spdi_to_rsid_cache_fpath}')
            with gzip.open(self.spdi_to_rsid_cache_fpath, 'rt') as cache_file:
                for line in cache_file:
                    splat = line.strip().split('\t')
                    if len(splat) > 1:
                        spdi, rsid = splat
                        if spdi not in self._spdi_to_rsid:
                            self._spdi_to_rsid[spdi] = set()
                        self._spdi_to_rsid[spdi].add(rsid)
                    else:
                        self._missing_spdis.add(splat[0])

        # Load rsID to SPDI cache
        if self.rsid_to_spdi_cache_fpath and self.rsid_to_spdi_cache_fpath.exists():
            logger.info(f'Loading cached rsid-to-spdi translations from {self.rsid_to_spdi_cache_fpath}')
            with gzip.open(self.rsid_to_spdi_cache_fpath, 'rt') as cache_file:
                for line in cache_file:
                    splat = line.strip().split('\t')
                    if len(splat) > 1:
                        rsid, spdi = splat
                        if rsid not in self._rsid_to_spdi:
                            self._rsid_to_spdi[rsid] = set()
                        self._rsid_to_spdi[rsid].add(spdi)
                    else:
                        self._missing_rsids.add(splat[0])

    def _update_spdi_to_rsid_cache_file(self, new_missing_spdis: Set[str], new_dict_items: Dict[str, Set[str]], mode: str = 'at') -> None:
        """Update SPDI to rsID cache file."""
        if not self.spdi_to_rsid_cache_fpath:
            return
            
        with gzip.open(self.spdi_to_rsid_cache_fpath, mode) as cache_file:
            for spdi in new_missing_spdis:
                cache_file.write(f'{spdi}\n')
            for spdi, rsids in new_dict_items.items():
                for rsid in rsids:
                    cache_file.write(f'{spdi}\t{rsid}\n')

    def _update_rsid_to_spdi_cache_file(self, new_missing_rsids: Set[str], new_dict_items: Dict[str, Set[str]], mode: str = 'at') -> None:
        """Update rsID to SPDI cache file."""
        if not self.rsid_to_spdi_cache_fpath:
            return
            
        with gzip.open(self.rsid_to_spdi_cache_fpath, mode) as cache_file:
            for rsid in new_missing_rsids:
                cache_file.write(f'{rsid}\n')
            for rsid, spdis in new_dict_items.items():
                for spdi in spdis:
                    cache_file.write(f'{rsid}\t{spdi}\n')

    def _rewrite_cache(self) -> None:
        """Rewrite both cache files from memory."""
        if self.spdi_to_rsid_cache_fpath:
            self._update_spdi_to_rsid_cache_file(
                new_missing_spdis=self._missing_spdis,
                new_dict_items=self._spdi_to_rsid,
                mode='wt'
            )
        if self.rsid_to_spdi_cache_fpath:
            self._update_rsid_to_spdi_cache_file(
                new_missing_rsids=self._missing_rsids,
                new_dict_items=self._rsid_to_spdi,
                mode='wt'
            )

    @abstractmethod
    def translate_spdis_to_rsids(self, spdis: Iterable[str], as_set: bool = False) -> Union[List[str], Set[str]]:
        """
        Translate SPDI identifiers to rsIDs.

        Args:
            spdis: SPDI identifiers to translate
            as_set: Return results as a set instead of list

        Returns:
            List or set of translated rsIDs
        """
        pass

    @abstractmethod 
    def translate_rsids_to_spdis(self, rsids: Iterable[str], as_set: bool = False) -> Union[List[str], Set[str]]:
        """
        Translate rsIDs to SPDI identifiers.

        Args:
            rsids: rsIDs to translate
            as_set: Return results as a set instead of list

        Returns:
            List or set of translated SPDIs
        """
        pass

    def get_translations_bidirectional(self, spdis: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get bidirectional mappings between SPDIs and rsIDs.

        Args:
            spdis: SPDI identifiers to translate

        Returns:
            Tuple of (spdi_to_rsid, rsid_to_spdi) dictionaries
        """
        spdi_to_rsid = {}
        rsid_to_spdi = {}

        for spdi in spdis:
            if spdi in self._spdi_to_rsid:
                rsid = sorted(self._spdi_to_rsid[spdi])[0]
                spdi_to_rsid[spdi] = rsid
                rsid_to_spdi[rsid] = spdi

        return spdi_to_rsid, rsid_to_spdi

    def _add_to_cache(self, spdi: str, rsid: str) -> None:
        """Add a single SPDI-rsID pair to both caches."""
        # Update in-memory caches
        if spdi not in self._spdi_to_rsid:
            self._spdi_to_rsid[spdi] = {rsid}
        else:
            self._spdi_to_rsid[spdi].add(rsid)
        
        if rsid not in self._rsid_to_spdi:
            self._rsid_to_spdi[rsid] = {spdi}
        else:
            self._rsid_to_spdi[rsid].add(spdi)

        # Update cache files
        if self.spdi_to_rsid_cache_fpath:
            with gzip.open(self.spdi_to_rsid_cache_fpath, 'at') as f:
                f.write(f"{spdi}\t{rsid}\n")
        if self.rsid_to_spdi_cache_fpath:
            with gzip.open(self.rsid_to_spdi_cache_fpath, 'at') as f:
                f.write(f"{rsid}\t{spdi}\n")


class SpdiRsidTranslatorDbSnp(SpdiRsidTranslator):
    """SPDI to rsID translator using dbSNP JSON files."""
    
    def __init__(
        self,
        dbsnp_vcf_fpath: Optional[Union[Path, str]] = None,
        cache_path: Optional[Union[Path, str]] = None, 
        threads: int = DEFAULT_THREADS,
        chrom_mapper: Optional[ChromMapper] = None,
        source_chrom_namespace: str = DBSNP_CHROM_NAMESPACE,
        dest_chrom_namespace: str = WORKING_CHROM_NAMESPACE,
        pos_offset: int = DBSNP_SPDI_POS_OFFSET
    ):
        """
        Initialize translator with either a dbSNP VCF file or existing map file.
        
        Args:
            dbsnp_vcf_fpath: Path to dbSNP VCF file
            cache_path: Directory for storing map and cache files
            threads: Number of threads to use for processing
            chrom_mapper: ChromMapper instance for chromosome name translation
            source_chrom_namespace: Source chromosome namespace
            dest_chrom_namespace: Destination chromosome namespace
            pos_offset: Offset to apply to dbSNP positions
        """
        logger.info(f'Initializing SPDI to RSID translator based on dbSNP data in {dbsnp_vcf_fpath} ...')
        cache_path = Path(cache_path) if cache_path else Path.cwd()

        self.dbsnp_vcf_fpath = Path(dbsnp_vcf_fpath) if dbsnp_vcf_fpath else None
        self.chrom_mapper = chrom_mapper
        self.source_chrom_namespace = source_chrom_namespace
        self.dest_chrom_namespace = dest_chrom_namespace
        self.spdi_map_fpath = cache_path.joinpath('spdi_to_rsid.txt.bgz')
        self.rsid_map_fpath = cache_path.joinpath('rsid_to_spdi.txt.bgz')
        self.pos_offset = pos_offset
        self.rsid_map_index_fpath = Path(str(self.rsid_map_fpath) + '.csi')
        self.threads = threads
        
        super().__init__(cache_path)

        spdi_map_index_fpath = Path(str(self.spdi_map_fpath) + '.csi')
        rsid_map_index_fpath = Path(str(self.rsid_map_fpath) + '.csi')

        # Generate map files if needed
        if self.dbsnp_vcf_fpath and not (self.spdi_map_fpath.exists() and self.rsid_map_fpath.exists() and spdi_map_index_fpath.exists() and rsid_map_index_fpath.exists()):
            logger.info('Generating SPDI to rsID map files from dbSNP VCF...')
            self.generate_spdi_to_rsid_map_files_from_dbsnp_vcf()
            
        if not self.spdi_map_fpath.exists():
            raise FileNotFoundError(f'SPDI map file not found: {self.spdi_map_fpath}')
        else:
            logger.info(f'SPDI map file found at {self.spdi_map_fpath}')
            
        if not self.rsid_map_fpath.exists():
            raise FileNotFoundError(f'rsID map file not found: {self.rsid_map_fpath}')
        else:
            logger.info(f'rsID map file found at {self.rsid_map_fpath}')
            
        # Verify map files are indexed
        if not spdi_map_index_fpath.exists():
            raise FileNotFoundError(f'SPDI map index not found: {spdi_map_index_fpath}')
        else:
            logger.info(f'SPDI map index found at {spdi_map_index_fpath}')

        if not rsid_map_index_fpath.exists():
            raise FileNotFoundError(f'rsID map index not found: {rsid_map_index_fpath}')
        else:
            logger.info(f'rsID map index found at {rsid_map_index_fpath}')
    
    def generate_spdi_to_rsid_map_files_from_dbsnp_vcf(self):
        # var_idx_to_rsid = {}
        row_chunk_size = 10000 # Write rows in chunks to improve compression efficiency (hopefully)

        with gzip.open(self.dbsnp_vcf_fpath, 'rt', encoding='utf-8') as vcf_file, gzip.open(self.spdi_map_fpath, 'wt') as spdi_to_rsid_file, gzip.open(self.rsid_map_fpath, 'wt') as rsid_to_spdi_file:
            csv_reader = csv.DictReader((row for row in vcf_file if not row.startswith('#')),
                                        delimiter='\t',
                                        quoting=csv.QUOTE_NONE,
                                        fieldnames=['CHROM','POS', 'ID',	'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])  
            
            spdi_to_rsid_row_buffer = []
            rsid_to_spdi_row_buffer = []
            if self.source_chrom_namespace != self.dest_chrom_namespace:
                chrom_dict = self.chrom_mapper.get_chrom_dict(self.source_chrom_namespace, self.dest_chrom_namespace)                
            else:
                chrom_dict = {}

            for line_num, row in tqdm(enumerate(csv_reader), 'dbSNP VCF lines'):
                rsid = row['ID']
                chrom = row['CHROM']
                if chrom_dict and chrom in chrom_dict:
                    chrom = chrom_dict[chrom]

                pos = int(row['POS'])
                ref = row['REF']
                alts = row['ALT']
                
                for alt in alts.split(','):
                    pos, ref, alt = right_shift_variant_for_vep(pos, ref, alt)
                    str_pos = str(pos + self.pos_offset)
                    
                    spdi = ':'.join((chrom, str_pos, ref, alt))
                    assert rsid.startswith('rs')
                    rs_num = rsid[2:]
                    
                    spdi_to_rsid_row_buffer.append('\t'.join((chrom, str_pos, spdi, rsid)))
                    rsid_to_spdi_row_buffer.append('\t'.join(('rs', rs_num, spdi)))

                    if len(spdi_to_rsid_row_buffer) >= row_chunk_size:
                        spdi_to_rsid_file.writelines(spdi_to_rsid_row_buffer)
                        rsid_to_spdi_file.writelines(rsid_to_spdi_row_buffer)
                        spdi_to_rsid_row_buffer = []
                        rsid_to_spdi_row_buffer = []
                
            spdi_to_rsid_file.writelines(spdi_to_rsid_row_buffer)
            rsid_to_spdi_file.writelines(rsid_to_spdi_row_buffer)

        # Sort spdi_to_rsid file by chrom and pos
        logger.info('Sorting spdi_to_rsid file...')
        sort_cmd = ['sort', '-k1,1', '-k2,2n', '-T', str(self.cache_path), self.spdi_map_fpath, '-o', self.spdi_map_fpath]
        subprocess.run(sort_cmd, check=True)

        # Sort rsid_to_spdi file by rs prefix and rs number
        logger.info('Sorting rsid_to_spdi file...')
        sort_cmd = ['sort', '-k1,1', '-k2,2n', '-T', str(self.cache_path), self.rsid_map_fpath, '-o', self.rsid_map_fpath]
        subprocess.run(sort_cmd, check=True)

        # Index spdi_to_rsid file
        logger.info('Indexing spdi_to_rsid file...')
        tabix_cmd = ['tabix', '-s', '1', '-b', '2', '-e', '2', '--csi', self.spdi_map_fpath]
        subprocess.run(tabix_cmd, check=True)

        # Index rsid_to_spdi file 
        logger.info('Indexing rsid_to_spdi file...')
        tabix_cmd = ['tabix', '-s', '1', '-b', '2', '-e', '2', '--csi', self.rsid_map_fpath]
        subprocess.run(tabix_cmd, check=True)

    @staticmethod
    def _open_file(fpath, mode='rt'):
        if Path(fpath).suffix == '.bz2':
            return bz2.open(fpath, mode)
        elif Path(fpath).suffix == '.gz':
            return gzip.open(fpath, mode)
        else:
            return open(fpath, mode)

    def _load_from_cache_file(self):
        with self._open_file(self.cache_file_fpath) as cache_file:
            for line in cache_file:
                splat = line.strip('\n').split('\t')
                if len(splat) == 1:
                    self._missing_spdis.add(splat[0])
                elif len(splat) == 2:
                    self._spdi_to_rsid[splat[0]] = splat[1:]

    def _update_cache_file(self, new_missing, new_dict_items, mode='at'):
        with self._open_file(self.spdi_to_rsid_cache_fpath, mode) as cache_file:
            for spdi in new_missing:
                cache_file.write(f'{spdi}\n')
            for k, v in new_dict_items.items():
                cache_file.write(f'{k}\t{"\t".join(v)}\n')

    def _rewrite_cache(self):
        self._update_cache_file(new_missing = self._missing_spdis, new_dict_items=self._spdi_to_rsid, mode='wt')
        
    def __contains__(self, spdi):
        return self[spdi] is not None

    def __getitem__(self, spdi):
        if spdi in self._spdi_to_rsid:
            return self._spdi_to_rsid[spdi]
        else:
            if spdi in self._missing_spdis:
                return ''
            else:
                # print(f'Doing single query for {spdi}, this is bad')
                new_dict = self._get_spdi_to_rsid_dict([spdi])
                if new_dict:
                    self._spdi_to_rsid.update(new_dict)
                    self._update_cache_file(new_missing = set([]), new_dict_items=new_dict)
                    return new_dict[spdi]
                else:
                    self.missing_spdis.add(spdi)
                    self._update_cache_file(new_missing = set([spdi]), new_dict_items={})
                    return ''

 
    def _query_spdis(self, spdis: List[str]):
        region_file_lines = []
        for spdi in spdis:
            chrom, pos, ref, alt = spdi.split(':')
            pos = int(pos) + 1
            region_file_lines.append(f'{chrom}\t{pos}')

        region_file_string = '\n'.join(region_file_lines)

        with tempfile.NamedTemporaryFile(mode='w+t',
                                         encoding='utf-8',
                                         delete=True) as region_file:
            region_file.write(region_file_string)
            region_file.flush()
            results = self._execute_tabix_regionfile_query(region_file.name)

        return results

    def _get_spdi_to_rsid_dict(self, spdis: Iterable[str], use_tqdm=False):
        spdi_to_rsid = {}
    
        for spdi, rsid in iterate_cols(self._query_spdis(spdis), ['spdi', 'rsid']):
            if spdi in spdis:
                if spdi not in spdi_to_rsid:
                    spdi_to_rsid[spdi] = set([rsid])
                else:
                    spdi_to_rsid[spdi].add(rsid)
            
        return spdi_to_rsid   
    
    def _execute_tabix_regionfile_query(self, region_file_fpath):
        # print('Running tabix query!')
        cmd = ['tabix', self.spdi_map_fpath, '-@', str(self.threads), '-R', str(region_file_fpath)]
        return pd.read_csv(io.StringIO(subprocess.check_output(cmd).decode()), sep='\t', names=['chrom', 'pos', 'spdi', 'rsid'])

    def populate_spdis(self, spdis: Iterable[str]) -> None:
        """
        Populate cache with translations for given SPDIs.
        
        Args:
            spdis: SPDI identifiers to translate
        """
        spdis = set(spdis)
        spdis_to_query = spdis.difference(self._spdi_to_rsid.keys()).difference(self._missing_spdis)

        if len(spdis_to_query):
            update_dict = self._get_spdi_to_rsid_dict(spdis_to_query)
            new_missing = spdis_to_query.difference(update_dict.keys())

            # Update both direction caches
            for spdi, rsids in update_dict.items():
                self._spdi_to_rsid[spdi] = rsids
                for rsid in rsids:
                    if rsid not in self._rsid_to_spdi:
                        self._rsid_to_spdi[rsid] = {spdi}
                    else:
                        self._rsid_to_spdi[rsid].add(spdi)

            self._missing_spdis.update(new_missing)
            
            # Update both cache files
            self._update_spdi_to_rsid_cache_file(new_missing_spdis=new_missing, new_dict_items=update_dict)
            
            # Create reverse mapping for rsid cache update
            rsid_update_dict = {}
            for spdi, rsids in update_dict.items():
                for rsid in rsids:
                    if rsid not in rsid_update_dict:
                        rsid_update_dict[rsid] = {spdi}
                    else:
                        rsid_update_dict[rsid].add(spdi)
            self._update_rsid_to_spdi_cache_file(new_missing_rsids=set(), new_dict_items=rsid_update_dict)

    def translate_spdis_to_rsids(
        self, 
        spdis: Iterable[str], 
        as_set: bool = False, 
        multi_rsid_handling: str = 'first'
    ) -> Union[List[str], Set[str]]:
        """
        Translate SPDI identifiers to rsIDs.

        Args:
            spdis: SPDI identifiers to translate
            as_set: Return results as a set instead of list
            multi_rsid_handling: How to handle multiple rsIDs for a single SPDI
                'first': Use only the first rsID (sorted alphabetically)
                'all': Include all rsIDs (only when as_set=True)
                'string': Join multiple rsIDs with commas

        Returns:
            List or set of translated rsIDs
        """
        self.populate_spdis(spdis)

        if as_set:
            rsids = set()
            for spdi in spdis:
                result = self._spdi_to_rsid.get(spdi, set())
                if result:
                    if len(result) == 1 or (len(result) > 1 and multi_rsid_handling == 'first'):
                        rsids.add(sorted(result)[0])
                    elif multi_rsid_handling == 'string':
                        rsids.add(','.join(sorted(result)))
                    else:
                        rsids.update(result)
            return rsids
        else:
            rsids = []
            for spdi in spdis:
                result = self._spdi_to_rsid.get(spdi, set())
                if not result:
                    rsids.append('')
                else:
                    if len(result) == 1 or (len(result) > 1 and multi_rsid_handling == 'first'):
                        rsids.append(sorted(result)[0])
                    elif multi_rsid_handling == 'string':
                        rsids.append(','.join(sorted(result)))
                    else:
                        rsids.append(result)
            return rsids

    def get_translations_bidirectional(self, spdis: Iterable[str]):
        """
        Return a one-to-one mapping of spdis to rsids and vice-versa, keeping the first rsid hit for each spdi
        """        
        self.populate_spdis(spdis)
        
        spdi_to_rsid = {}
        rsid_to_spdi = {}

        for spdi in spdis:
            if spdi in self._spdi_to_rsid:
                rsid = sorted(self._spdi_to_rsid[spdi])[0]
                spdi_to_rsid[spdi] = rsid
                rsid_to_spdi[rsid] = spdi

        return spdi_to_rsid, rsid_to_spdi        

    def translate_rsids_to_spdis(
        self, 
        rsids: Iterable[str], 
        as_set: bool = False, 
        multi_spdi_handling: str = 'first'
    ) -> Union[List[str], Set[str]]:
        """
        Translate rsIDs to SPDIs.
        
        Args:
            rsids: Iterable of rsIDs to translate
            as_set: Return results as a set instead of list
            multi_spdi_handling: How to handle multiple SPDIs for a single rsID
                'first': Use only the first SPDI (sorted alphabetically)
                'all': Include all SPDIs (only when as_set=True)
                'string': Join multiple SPDIs with commas
                
        Returns:
            List or set of translated SPDI identifiers
        """
        self.populate_rsids(rsids)

        if as_set:
            spdis = set()
            for rsid in rsids:
                result = self._rsid_to_spdi.get(rsid, set())
                if result:
                    if len(result) == 1 or (len(result) > 1 and multi_spdi_handling == 'first'):
                        spdis.add(sorted(result)[0])
                    elif multi_spdi_handling == 'string':
                        spdis.add(','.join(sorted(result)))
                    else:
                        spdis.update(result)
            return spdis
        else:
            spdis = []
            for rsid in rsids:
                result = self._rsid_to_spdi.get(rsid, set())
                if not result:
                    spdis.append('')
                else:
                    if len(result) == 1 or (len(result) > 1 and multi_spdi_handling == 'first'):
                        spdis.append(sorted(result)[0])
                    elif multi_spdi_handling == 'string':
                        spdis.append(','.join(sorted(result)))
                    else:
                        spdis.append(result)
            return spdis

    def populate_rsids(self, rsids: Iterable[str]) -> None:
        """
        Populate cache with translations for given rsIDs.
        
        Args:
            rsids: rsIDs to translate
        """
        rsids = set(rsids)
        rsids_to_query = rsids.difference(self._rsid_to_spdi.keys()).difference(self._missing_rsids)

        if len(rsids_to_query):
            update_dict = self._get_rsid_to_spdi_dict(rsids_to_query)
            new_missing = rsids_to_query.difference(update_dict.keys())

            # Update both direction caches
            for rsid, spdis in update_dict.items():
                self._rsid_to_spdi[rsid] = spdis
                for spdi in spdis:
                    if spdi not in self._spdi_to_rsid:
                        self._spdi_to_rsid[spdi] = {rsid}
                    else:
                        self._spdi_to_rsid[spdi].add(rsid)

            self._missing_rsids.update(new_missing)
            
            # Update both cache files
            self._update_rsid_to_spdi_cache_file(new_missing_rsids=new_missing, new_dict_items=update_dict)
            
            # Create reverse mapping for spdi cache update
            spdi_update_dict = {}
            for rsid, spdis in update_dict.items():
                for spdi in spdis:
                    if spdi not in spdi_update_dict:
                        spdi_update_dict[spdi] = {rsid}
                    else:
                        spdi_update_dict[spdi].add(rsid)
            self._update_spdi_to_rsid_cache_file(new_missing_spdis=set(), new_dict_items=spdi_update_dict)

    def _get_rsid_to_spdi_dict(self, rsids: Iterable[str]) -> Dict[str, Set[str]]:
        """
        Get dictionary mapping rsIDs to sets of SPDIs.
        
        Args:
            rsids: rsIDs to query
            
        Returns:
            Dictionary mapping rsIDs to sets of SPDIs
        """
        rsid_to_spdi = {}
        
        results = self._query_rsids(list(rsids))
        for rsid, spdi in iterate_cols(results, ['rsid', 'spdi']):
            if rsid in rsids:
                if rsid not in rsid_to_spdi:
                    rsid_to_spdi[rsid] = {spdi}
                else:
                    rsid_to_spdi[rsid].add(spdi)
            
        return rsid_to_spdi

    def _query_rsids(self, rsids: List[str]) -> pd.DataFrame:
        """
        Query rsIDs from the mapping file.
        
        Args:
            rsids: List of rsIDs to query
            
        Returns:
            DataFrame with rsid and spdi columns
        """
        region_file_lines = []
        for rsid in rsids:
            region_file_lines.append(rsid)

        region_file_string = '\n'.join(region_file_lines)

        with tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', delete=True) as region_file:
            region_file.write(region_file_string)
            region_file.flush()
            
            cmd = ['tabix', self.rsid_map_fpath, '-@', str(self.threads), '-R', str(region_file.name)]
            return pd.read_csv(
                io.StringIO(subprocess.check_output(cmd).decode()), 
                sep='\t', 
                names=['rsid', 'spdi']
            )


def initialize_spdi_translator(
    config: MRConfig,
    plink_wrapper: PlinkWrapper,
    chrom_mapper: ChromMapper,
    source_chrom_namespace: str = DBSNP_CHROM_NAMESPACE,
    dest_chrom_namespace: str = WORKING_CHROM_NAMESPACE
) -> SpdiRsidTranslator:
    """
    Initialize appropriate SPDI translator based on configuration.

    Args:
        config: MR configuration object
        plink_wrapper: Initialized PlinkWrapper instance
        chrom_mapper: Initialized ChromMapper instance
        source_chrom_namespace: Source chromosome namespace
        dest_chrom_namespace: Destination chromosome namespace

    Returns:
        Initialized SpdiRsidTranslator instance
    """
    if config.dbsnp_vcf_fpath:      
        return SpdiRsidTranslatorDbSnp(
            dbsnp_vcf_fpath=config.dbsnp_vcf_fpath,
            cache_path=config.cache_path,
            threads=config.threads,
            chrom_mapper=chrom_mapper,
            source_chrom_namespace=source_chrom_namespace,
            dest_chrom_namespace=dest_chrom_namespace,
            pos_offset=DBSNP_SPDI_POS_OFFSET
        )
  
    else:
        return SpdiRsidTranslator1kg(
            plink_wrapper=plink_wrapper,
            chrom_mapper=chrom_mapper,
            cache_path=config.cache_path,
            spdi_pos_offset=THOUSAND_GENOMES_POS_OFFSET
        )

class SpdiRsidTranslator1kg(SpdiRsidTranslator):
    """
    SPDI to rsID translator using 1000 Genomes data.
    Uses PLINK bim files for translations.
    """
    def __init__(
        self, 
        plink_wrapper: PlinkWrapper,
        chrom_mapper: ChromMapper,
        cache_path: Optional[Union[Path, str]] = None,
        spdi_pos_offset: int = THOUSAND_GENOMES_POS_OFFSET
    ):
        """
        Initialize 1000 Genomes translator.

        Args:
            plink_wrapper: PlinkWrapper instance for handling PLINK operations
            chrom_mapper: ChromMapper instance for chromosome name translation
            cache_path: Optional path to cache directory
            spdi_pos_offset: Position offset for SPDI coordinates
        """
        super().__init__(cache_path)
        self.plink_wrapper = plink_wrapper
        self.chrom_mapper = chrom_mapper
        self.spdi_pos_offset = spdi_pos_offset
        self.load_from_bim()

    def load_from_bim(self) -> None:
        """Load translations from PLINK bim file."""
        for rsid, chrom, pos, ref, alt in self.plink_wrapper.iterate_bim_variants():
            this_spdi = (
                f'{self.chrom_mapper.translate_chrom_name(chrom=chrom, source_namespace="plain", dest_namespace="ucsc")}'
                f':{int(pos) + self.spdi_pos_offset}:{ref}:{alt}'
            )
            
            if this_spdi not in self._spdi_to_rsid:
                self._spdi_to_rsid[this_spdi] = {rsid}
            else:
                self._spdi_to_rsid[this_spdi].add(rsid)
                
            if rsid not in self._rsid_to_spdi:
                self._rsid_to_spdi[rsid] = {this_spdi}
            else:
                self._rsid_to_spdi[rsid].add(this_spdi)



