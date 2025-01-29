import logging
import pandas as pd
import subprocess
import tempfile
import warnings
import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, Collection
from glob import glob

from mr_miner.constants import (
    PLINK_FPATH,
    DEFAULT_POP,
    DEFAULT_PLINK_THREADS,
    DEFAULT_PROXY_R2_THRESHOLD,
    DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
)
from mr_miner.utilities import iterate_cols

logger = logging.getLogger(__name__)


class PlinkWrapper:
    """
    Note: We don't currently have a solid set of binary files for GRCh38 that we can use confidently, so for now we do spdi-to-rsid translation using a GRCh38 dbSNP build
    (because SPDIs are build-dependent) but get the correlation data, clump and find proxies using the GRCh37 1kg data (since rsids are build-independent).
    """

    def __init__(self, data_1kg_prefix_template: str, pop: str = DEFAULT_POP, plink_fpath: str = PLINK_FPATH) -> None:
        self.plink_fpath = Path(plink_fpath)
        self.data_fpath_template = Path(data_1kg_prefix_template)
        self.pop = pop
        self.bim_file_prefix = Path(f'{str(self.data_fpath_template)}').with_name(f'{self.data_fpath_template.name}'.format(pop=pop))
        self.bim_file_fpath = self.bim_file_prefix.with_suffix('.bim')
        self.valid_rsids: set[str] = set()

        self._load_bim_file()

    def get_corr_matrix(
        self,
        rsids: Iterable[str],
        threads: int = DEFAULT_PLINK_THREADS,
        delete_temp_files: bool = True,
    ) -> pd.DataFrame:
        # Since we're no longer using plink to pre-process the rsid list into a BIM subset, we need to do it ourselves by first
        # intersecting the rsid list with the BIM.
        rsids = set(rsids).intersection(self.valid_rsids)
        if not len(rsids):
            warnings.warn("No valid rsids remain in LD correlation query.")
            return pd.DataFrame()

        with (
            tempfile.NamedTemporaryFile(
                "wt", delete=delete_temp_files, delete_on_close=delete_temp_files
            ) as snp_file,
            tempfile.NamedTemporaryFile(
                "rt", delete=delete_temp_files, delete_on_close=delete_temp_files
            ) as matrix_file,
        ):
            snp_file.write("\n".join(rsids))
            snp_file.flush()

            generate_matrix_cmd = [
                str(self.plink_fpath),
                "--bfile",
                str(self.bim_file_prefix),
                "--extract",
                snp_file.name,
                "--r",
                "square",
                "--threads",
                str(threads),
                "--keep-allele-order",
                "--out",
                matrix_file.name,
            ]
            try:
                subprocess.check_output(
                    generate_matrix_cmd, stderr=subprocess.STDOUT
                ).decode()
            except subprocess.CalledProcessError as cpe:
                print(cpe.cmd)
                print(cpe.returncode)
                print(cpe.output.decode())
                raise (cpe)

            ld_matrix = pd.read_csv(matrix_file.name + ".ld", sep="\t", header=None)

            assert ld_matrix.shape[0] == ld_matrix.shape[1] == len(rsids)
            ld_matrix.index = rsids
            ld_matrix.columns = rsids

        if delete_temp_files:
            for fpath in glob(matrix_file.name + "*"):
                os.remove(fpath)

        return ld_matrix

    def get_region_corr_matrix(
        self, 
        region: Tuple[str, int, int], 
        threads: int = DEFAULT_PLINK_THREADS
    ) -> pd.DataFrame:
        with (
            tempfile.NamedTemporaryFile("wt", delete_on_close=True) as region_file,
            tempfile.NamedTemporaryFile("rt", delete_on_close=True) as matrix_file,
        ):
            # For now just do a single region to avoid having to disentangle chromosomes
            # for i, (chrom, start, end) in enumerate(region_list):
            #     region_file.write(f'{chrom}\t{start}\t{end}\tR{i}\n')
            chrom, start, end = region
            i = "R0"
            region_file.write(f"{chrom}\t{start}\t{end}\tR{i}\n")
            region_file.flush()

            generate_bim_file_cmd = [
                str(self.plink_fpath),
                "--bfile",
                str(self.bim_file_prefix),
                "--extract",
                "range",
                region_file.name,
                "--make-just-bim",
                "--threads",
                str(threads),
                "--keep-allele-order",
                "--out",
                matrix_file.name,
            ]

            try:
                subprocess.check_output(
                    generate_bim_file_cmd, stderr=subprocess.STDOUT
                ).decode()
            except subprocess.CalledProcessError as cpe:
                print(cpe.cmd)
                print(cpe.returncode)
                print(cpe.output.decode())
                raise (cpe)

            bim_df = pd.read_csv(
                matrix_file.name + ".bim",
                sep="\t",
                names=["chrom", "rsid", "phenotype", "pos", "ref", "alt"],
            )
            assert (
                len(bim_df.chrom.unique()) == 1
            ), f'Plink query for region {region} returned multiple chromosomes: {",".join(bim_df.chrom.unique())}'

            generate_matrix_cmd = [
                str(self.plink_fpath),
                "--bfile",
                str(self.bim_file_prefix),
                "--extract",
                matrix_file.name + ".bim",
                "--r",
                "square",
                "--threads",
                str(threads),
                "--keep-allele-order",
                "--out",
                matrix_file.name,
            ]
            try:
                subprocess.check_output(
                    generate_matrix_cmd, stderr=subprocess.STDOUT
                ).decode()
            except subprocess.CalledProcessError as cpe:
                print(cpe.cmd)
                print(cpe.returncode)
                print(cpe.output.decode())
                raise (cpe)

            ld_matrix = pd.read_csv(matrix_file.name + ".ld", sep="\t", header=None)

            rsids_in_region = bim_df.rsid.tolist()
            assert ld_matrix.shape[0] == ld_matrix.shape[1] == len(rsids_in_region)
            ld_matrix.index = rsids_in_region
            ld_matrix.columns = rsids_in_region

        return ld_matrix

    def _load_bim_file(self) -> None:
        if not self.bim_file_fpath.exists():
            logger.error(f'BIM file not found at {self.bim_file_fpath}')
            raise FileNotFoundError(f'BIM file not found at {self.bim_file_fpath}')

        bim_df = pd.read_csv(
            self.bim_file_fpath,
            sep='\t',
            header=None,
            names=['chrom', 'snp_id', 'cm', 'pos', 'allele1', 'allele2'],
        )
        self.valid_rsids = set(bim_df.snp_id)

    def get_proxy_variants(
        self, 
        query_rsids: Iterable[str], 
        r2_threshold: float = DEFAULT_PROXY_R2_THRESHOLD,
        distance_kbp: float = DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
        threads: int = DEFAULT_PLINK_THREADS, 
        verbose: bool = False, 
        no_delete: bool = False
    ) -> Dict[str, list[str]]:
        with tempfile.NamedTemporaryFile(
            mode='wt',
            delete=not no_delete,
            delete_on_close=not no_delete
        ) as snp_file, tempfile.NamedTemporaryFile(
            mode='rt',
            delete=not no_delete,
            delete_on_close=not no_delete
        ) as proxy_output_file:
            snp_file.write('\n'.join(query_rsids))
            snp_file.flush()
            
            generate_proxies_cmd = [str(self.plink_fpath),
                                     '--bfile', str(self.bim_file_prefix),
                                     '--show-tags', snp_file.name,
                                     '--tag-r2', str(r2_threshold),
                                     '--tag-kb', str(distance_kbp),
                                     '--list-all',
                                     '--threads', str(threads),
                                     '--out', proxy_output_file.name]
            try:
                proxy_output = subprocess.check_output(generate_proxies_cmd, stderr=subprocess.STDOUT).decode()
            except subprocess.CalledProcessError as cpe:
                print(cpe.cmd)
                print(cpe.returncode)
                print(cpe.output.decode())
                raise(cpe)
            else:
                if verbose:
                    print(proxy_output)
                
            proxy_results = pd.read_csv(proxy_output_file.name + '.tags.list', sep=r'\s+')
        
        return {target_rsid:tags.split('|') for target_rsid, tags in iterate_cols(proxy_results, ('SNP', 'TAGS')) if tags != 'NONE'}

    def iterate_bim_variants(self) -> Iterable[Tuple[str, str, int, str, str]]:
        bim_df = pd.read_csv(
            self.bim_file_fpath,
            sep="\t",
            header=None,
            names=["chrom", "snp_id", "cm", "pos", "allele1", "allele2"],
        )
        for _, row in bim_df.iterrows():
            yield (
                row.snp_id,
                row.chrom,
                row.pos,
                row.allele1,
                row.allele2,
            )
