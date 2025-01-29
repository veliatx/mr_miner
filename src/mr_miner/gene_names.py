import numpy as np
import pandas as pd
from pathlib import Path

from mr_miner.utilities import iterate_cols


class EnsemblMapper:
    def __init__(self, ensembl_gff_fpath: Path | str):
        """
        Initialize EnsemblMapper with path to Ensembl GFF file
        
        Args:
            ensembl_gff_fpath: Path to Ensembl GFF annotation file
        """
        self.ensembl_gff_fpath = Path(ensembl_gff_fpath)
        self._ensembl_to_name = {}
        self._name_to_ensembl = {}

        print(f'Loading gene name translations from Ensembl GFF at {self.ensembl_gff_fpath} ...')
        ensembl_gff = pd.read_csv(
            self.ensembl_gff_fpath,
            sep='\t',
            comment='#',
            low_memory=False,
            header=None
        )
        for row_idx, region_type, annotations in iterate_cols(ensembl_gff, [2, 8], preface_with_index=True):
            if region_type == 'gene':
                anno_dict = dict([pair.split('=') for pair in annotations.split(';')])
                if 'Name' not in anno_dict:
                    self._ensembl_to_name[anno_dict['gene_id']] = ''
                else:
                    self._ensembl_to_name[anno_dict['gene_id']] = anno_dict['Name']
                    self._name_to_ensembl[anno_dict['Name']] = anno_dict['gene_id']

    def convert_ensembl_to_name(self, ensembl_gene_id: str) -> str:
        if '.' in ensembl_gene_id:
            ensembl_gene_id = ensembl_gene_id.split('.')[0]
        if ensembl_gene_id not in self._ensembl_to_name:
            return ''
        return self._ensembl_to_name[ensembl_gene_id]

    def convert_name_to_ensembl(self, gene_name: str) -> str:
        if gene_name not in self._name_to_ensembl:
            return ''
        return self._name_to_ensembl[gene_name]


class UniprotMapper:
    UNIPROT_COLUMN_NAMES = (
        'UniProtKB-AC',
        'UniProtKB-ID',
        'GeneID (EntrezGene)',
        'RefSeq',
        'GI',
        'PDB',
        'GO',
        'UniRef100',
        'UniRef90',
        'UniRef50',
        'UniParc',
        'PIR',
        'NCBI-taxon',
        'MIM',
        'UniGene',
        'PubMed',
        'EMBL',
        'EMBL-CDS',
        'Ensembl',
        'Ensembl_TRS',
        'Ensembl_PRO',
        'Additional PubMed'
    )

    def __init__(self, uniprot_mapping_table_fpath: Path | str):
        """
        Initialize UniProt ID mapper
        
        Args:
            uniprot_mapping_table_fpath: Path to UniProt ID mapping table
        """
        self.uniprot_mapping_table_fpath_fpath = Path(uniprot_mapping_table_fpath)
        self._incoming_translation_dict = {}
        self._outgoing_translation_dict = {}
        self.valid_identifier_types = set()

        self._load_mapping_table()

    def _load_mapping_table(self):
        self.valid_identifier_types = set(self.UNIPROT_COLUMN_NAMES)
        print(f'Loading gene name mapping data from {self.uniprot_mapping_table_fpath_fpath} ...')
        uniprot_mapping_table = pd.read_csv(
            self.uniprot_mapping_table_fpath_fpath,
            sep='\t',
            low_memory=False,
            header=None,
            names=self.UNIPROT_COLUMN_NAMES,
            dtype=str
        )

        self._incoming_translation_dict = {}
        self._outgoing_translation_dict = {}

        for col in uniprot_mapping_table.columns:
            self._incoming_translation_dict[col] = {}
            for i, val in enumerate(uniprot_mapping_table[col]):
                if val == '' or (isinstance(val, float) and np.isnan(val)):
                    continue
                for sub_val in val.split(';'):
                    if sub_val not in self._incoming_translation_dict[col]:
                        self._incoming_translation_dict[col][sub_val] = set()
                    self._incoming_translation_dict[col][sub_val].add(i)

            self._outgoing_translation_dict[col] = list(uniprot_mapping_table[col])

    def translate(
        self,
        source_gene_identifier: str,
        source_identifier_type: str,
        destination_identifier_type: str
    ) -> list:
        if source_identifier_type not in self.valid_identifier_types:
            raise ValueError(
                f'Invalid source gene identifier type {source_identifier_type} given. '
                f'Valid types are {", ".join(self.valid_identifier_types)}!'
            )
        if destination_identifier_type not in self.valid_identifier_types:
            raise ValueError(
                f'Invalid destination gene identifier type {destination_identifier_type} given. '
                f'Valid types are {", ".join(self.valid_identifier_types)}!'
            )

        if source_gene_identifier not in self._incoming_translation_dict[source_identifier_type]:
            return []

        identifier_indices = self._incoming_translation_dict[source_identifier_type][source_gene_identifier]
        translated_identifiers = set()
        for id_idx in identifier_indices:
            translated_id_string = self._outgoing_translation_dict[destination_identifier_type][id_idx]
            if translated_id_string == '' or (isinstance(translated_id_string, float) and np.isnan(translated_id_string)):
                continue

            translated_identifiers.update(translated_id_string.split(';'))

        return sorted(translated_identifiers)