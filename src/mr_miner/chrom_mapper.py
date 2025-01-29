"""Chromosome mapping utilities."""
import collections
from pathlib import Path
import logging
from typing import Any, Dict

import pandas as pd
import requests

from mr_miner.constants import NCBI_GRCH38_ASSEMBLY_REPORT_URL

logger = logging.getLogger(__name__)

class ChromMapper:
    """
    Wrapper around a couple of nested dicts giving chrom name translations and lengths.
    Can be initialized from a TSV file or generated from NCBI data.
    """

    def __init__(
        self,
        mapping_tsv_fpath: Path | str | None = None,
        assembly_report_url: str | None = NCBI_GRCH38_ASSEMBLY_REPORT_URL,
        allow_duplicates: bool = True,
    ) -> None:
        """
        Initialize ChromMapper.

        Args:
            mapping_tsv_fpath: Path to chromosome mapping TSV file
            assembly_report_url: URL to NCBI assembly report (if generating mappings)
            allow_duplicates: Whether to allow duplicate chromosome entries
        """
        self.assembly_report_url = assembly_report_url
        self.chroms: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)

        if not mapping_tsv_fpath.exists():
            logger.info(f"No chromosome mapping file found at {mapping_tsv_fpath}. Generating new one.")
            self.generate_mapping_file(mapping_tsv_fpath)
    
        self._load_mapping_file(
            mapping_tsv_fpath=mapping_tsv_fpath,
            allow_duplicates=allow_duplicates
        )

    def _load_mapping_file(
        self,
        mapping_tsv_fpath: Path | str,
        allow_duplicates: bool = True
    ) -> None:
        """Load chromosome mappings from TSV file."""
        logger.info(f"Loading chromosome mappings from {mapping_tsv_fpath} ...")
        mapping_df = pd.read_csv(mapping_tsv_fpath, sep="\t")

        for line_num, chrom_row in mapping_df.iterrows():
            this_chrom_dict = {
                "length": int(chrom_row['length'])
            }
            for assembly_col in mapping_df.columns:
                if assembly_col == 'length':
                    continue
                field_value = str(chrom_row[assembly_col]).strip()
                if pd.isna(field_value) or field_value == '':
                    continue
                if assembly_col not in this_chrom_dict:
                    this_chrom_dict[assembly_col] = field_value
                if (
                    field_value not in self.chroms[assembly_col]
                ):  # For duplicate identifiers, keep the mappings from the first such identifier encountered
                    this_chrom_dict[assembly_col] = str(field_value)
                    self.chroms[assembly_col][field_value] = this_chrom_dict
                elif not allow_duplicates:
                    raise ValueError(
                        f"Encountered a duplicate entry for chromosome {field_value} in namespace {assembly_col} on line {line_num+1} and `allow_duplicates` is set to False!\n{chrom_row}"
                    )
                    
    @staticmethod
    def _parse_assembly_report(content: str) -> pd.DataFrame:
        """
        Parse NCBI assembly report content into a DataFrame.
        
        Args:
            content: Raw content of assembly report
            
        Returns:
            DataFrame with chromosome information
        """
        from io import StringIO
        df = pd.read_csv(StringIO(content), sep="\t", comment='#', header=None, 
                         names = ['Sequence-Name',
                                        'Sequence-Role',
                                        'Assigned-Molecule',
                                        'Assigned-Molecule-Location/Type',
                                        'GenBank-Accn',
                                        'Relationship',
                                        'RefSeq-Accn',
                                        'Assembly-Unit',
                                        'Sequence-Length',
                                        'UCSC-style-name'
                                        ]   )
        logger.info(f"Parsed assembly report into DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
        # Rename columns to match expected format
        column_mapping = {
            'Sequence-Name': 'plain',
            'GenBank-Accn': 'genbank',
            'RefSeq-Accn': 'refseq',
            'UCSC-style-name': 'ucsc',
            'Sequence-Length': 'length'
        }
        return df.loc[:, list(column_mapping.keys())].rename(columns=column_mapping)
        
    def generate_mapping_file(
        self,
        output_path: Path | str,
    ) -> None:
        """
        Generate chromosome mapping TSV file from NCBI assembly report.
        
        Args:
            output_path: Where to save the mapping file
            assembly_report_url: URL to fetch the assembly report from
        """
        logger.info(f"Fetching assembly report from {self.assembly_report_url} ...")
        response = requests.get(self.assembly_report_url)
        response.raise_for_status()
        assembly_content = response.text
        df = self._parse_assembly_report(assembly_content)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Chromosome mapping file saved to {output_path}")
        
    def translate(self, chrom_name: str, from_namespace: str, to_namespace: str) -> str | None:
        """
        Translate a chromosome name from one namespace to another.
        
        Args:
            chrom_name: Chromosome name to translate
            from_namespace: Source namespace
            to_namespace: Target namespace
            
        Returns:
            Translated chromosome name or None if translation not found
        """
        if chrom_name not in self.chroms[from_namespace]:
            return None
        
        chrom_dict = self.chroms[from_namespace][chrom_name]
        if to_namespace not in chrom_dict:
            return None
            
        return chrom_dict[to_namespace]

    def get_length(self, chrom_name: str, namespace: str = "ucsc") -> int | None:
        """
        Get the length of a chromosome.
        
        Args:
            chrom_name: Chromosome name
            namespace: Namespace of the chromosome name
            
        Returns:
            Chromosome length in bases or None if not found
        """
        if chrom_name not in self.chroms[namespace]:
            return None
            
        return self.chroms[namespace][chrom_name]["length"]
    
    def get_chrom_dict(self, source_namespace: str, dest_namespace: str) -> dict[str, Dict[str, Any]]:
        """Get dictionary of translations from one namespace to another"""
        return {chrom: self.chroms[source_namespace][chrom][dest_namespace] for chrom in self.chroms[source_namespace]}
    
    def get_lengths(self, namespace: str) -> dict[str, int]:
        """Get dictionary of sequence lengths for all chromosomes in a namespace."""
        return {
            seq_name: self.chroms[namespace][seq_name]["length"]
            for seq_name in self.chroms[namespace]
        }

    @property
    def namespaces(self) -> list[str]:
        """Get sorted list of available namespaces."""
        return sorted(self.chroms.keys())
