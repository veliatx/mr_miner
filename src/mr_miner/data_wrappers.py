import collections
import gzip
import json
import logging
import subprocess
import io
import re
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from typing import Dict, Iterable, Any, Self, List, Set
from tqdm import tqdm

from mr_miner.constants import DEFAULT_THREADS, DEFAULT_TRAIT_COLUMN, Z_95
from mr_miner.variant_correlations import VariantCorrelations
from mr_miner.variant_ids import SpdiRsidTranslatorDbSnp
from mr_miner.utilities import (
    convert_dtypes_correctly,
    iterate_cols,
    my_in1d,
    trim_ensembl_gene,
    convert_gtex_variant_id_to_spdi
)
from mr_miner.variant_proxies import VariantProxies

logger = logging.getLogger(__name__)


def load_jsonl_as_df(json_fpath: Path | str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame.
    
    Args:
        json_fpath: Path to JSONL file
        
    Returns:
        DataFrame containing JSONL data
    """
    lines = []
    with open(json_fpath, 'rt') as jsonl_file:
        for line in jsonl_file:
            lines.append(json.loads(line.strip()))
    return pd.DataFrame(lines)


class Genome:
    def __init__(
        self,
        fasta_fpath: Path | str,
        chrom_mapper: Any,  # TODO: Add proper type hint for chrom_mapper
        source_chrom_style: str = "",
        dest_chrom_style: str = "",
        trim_seq_header_at_whitespace: bool = True,
        seq_name_keep_pattern: re.Pattern = re.compile(r"chr.+"),
    ) -> None:
        """
        Initialize Genome with path to FASTA file

        Args:
            fasta_fpath: Path to genome FASTA file
            chrom_mapper: Chromosome name mapper object
            source_chrom_style: Source chromosome naming style
            dest_chrom_style: Destination chromosome naming style
            trim_seq_header_at_whitespace: Whether to trim sequence headers at whitespace
            seq_name_keep_pattern: Regex pattern for sequence names to keep
        """
        self.fasta_fpath = Path(fasta_fpath)
        self.chrom_mapper = chrom_mapper
        self.trim_seq_header_at_whitespace = trim_seq_header_at_whitespace
        self.seq_name_keep_pattern = seq_name_keep_pattern
        self.seqs: Dict[str, str] = {}

        self.load_from_fasta()

        if source_chrom_style and dest_chrom_style:
            self.convert_chroms(source_chrom_style, dest_chrom_style)

    def load_from_fasta(self) -> None:
        """Load sequences from FASTA file."""
        seq_name = ""
        cur_seq_lines = []

        if self.fasta_fpath.suffix == ".gz":
            opener = gzip.open
        else:
            opener = open

        def update_seqs(seq_name, cur_seq_lines):
            if seq_name:
                if (
                    not self.seq_name_keep_pattern
                    or re.match(self.seq_name_keep_pattern, seq_name) is not None
                ):
                    self.seqs[seq_name] = "".join(cur_seq_lines)

        print(f"Loading genome sequences from {self.fasta_fpath} ...")
        with opener(self.fasta_fpath, "rt") as fasta_file:
            for line_num, line in tqdm(enumerate(fasta_file), desc="FASTA line"):
                line = line.strip()
                if line.startswith(">"):
                    update_seqs(seq_name, cur_seq_lines)

                    if self.trim_seq_header_at_whitespace:
                        seq_name = line[1:].split()[0]
                    else:
                        seq_name = line[1:]

                else:
                    cur_seq_lines.append(line)

        update_seqs(seq_name, cur_seq_lines)

        print(f"Loaded {len(self.seqs)} sequences.")

    def convert_chroms(self, source_chrom_style, dest_chrom_style):
        new_seqs: Dict[str, str] = {}
        for seq_name, seq in self.seqs.items():
            new_name = self.chrom_mapper.translate_chrom_name(
                chrom_name=seq_name,
                source_namespace=source_chrom_style,
                dest_namespace=dest_chrom_style,
            )
            new_seqs[new_name] = seq
        self.seqs = new_seqs

    def check_variant(self, chrom, pos_0based, ref):
        return self.seqs[chrom][pos_0based : pos_0based + len(ref)] == ref


# To Do: Integrate this into an outcomedata class for MR
class TabixedGenebassVariants:
    def __init__(
        self,
        genebass_path: Path | str = "/home/ubuntu/workspace/velia-analyses-dev/VAP_20240222_stats_gen/pipeline_runs/240620/intermediates/genebass/results_by_chrom/",
        glob_pattern: str = "genebass_variant_info_*.tsv.bgz",
    ) -> None:
        self.fpaths_by_chrom: Dict[str, Path] = {}
        genebass_path = Path(genebass_path)
        for fpath in genebass_path.glob(glob_pattern):
            this_chrom = fpath.stem.split('.')[0].split('_')[-1]
            self.fpaths_by_chrom[this_chrom] = fpath

    @property
    def available_chromosomes(self) -> List[str]:
        """Get list of available chromosomes sorted alphabetically."""
        return sorted(self.fpaths_by_chrom.keys())

    def query(
        self, 
        chrom: str, 
        start: int, 
        end: int, 
        num_threads: int = DEFAULT_THREADS
    ) -> pd.DataFrame:
        """Query variants in a genomic region.
        
        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            num_threads: Number of threads for tabix
            
        Returns:
            DataFrame with variant information
            
        Raises:
            KeyError: If chromosome not found
            subprocess.CalledProcessError: If tabix command fails
        """
        if chrom not in self.fpaths_by_chrom:
            raise KeyError(f'Chromosome {chrom} not found in available chromosomes')
            
        this_chrom_fpath = self.fpaths_by_chrom[chrom]
        
        # Get header from first line
        with gzip.open(this_chrom_fpath, 'rt') as gzip_file:
            col_names = next(gzip_file).strip().split('\t')

        tabix_cmd = [
            'tabix',
            '-@',
            str(num_threads),
            str(this_chrom_fpath),
            f'{chrom}:{start}-{end}',
        ]
        
        try:
            output = subprocess.check_output(tabix_cmd).decode()
        except subprocess.CalledProcessError as e:
            logger.error(f'Tabix query failed: {e}')
            raise

        return pd.read_csv(
            io.StringIO(output), 
            sep='\t', 
            names=col_names
        )


class ExposureData:
    """Base class for exposure data sources."""
    def __init__(self) -> None:
        super().__init__()


class GtexEqtls(ExposureData):
    def __init__(
        self, 
        gtex_eqtl_path: Path | str, 
        spdi_translator: SpdiRsidTranslatorDbSnp | None = None
    ) -> None:
        self.gtex_path = Path(gtex_eqtl_path)
        self.spdi_translator = spdi_translator

        super().__init__()

        self.eqtl_data: Dict[str, pd.DataFrame] = {}
        self._row_ids_by_variant: Dict[str, Dict[str, List[int]]] = {}
        self._row_ids_by_gene: Dict[str, Dict[str, List[int]]] = {}
        self._chroms_by_gene: Dict[str, Dict[str, Set[str]]] = {}
        self.variants_by_gene: Dict[str, Dict[str, Set[str]]] = {}
        self.all_spdis: Set[str] = set()
        self.all_genes: Set[str] = set()

        # Load tissue paths
        self.tissue_paths: Dict[str, Path] = {}
        for this_tissue_fpath in tqdm(
            list(self.gtex_path.glob('*.signif_pairs.*')), 
            desc='Tissue'
        ):
            tissue_name = this_tissue_fpath.stem.split('.')[0]
            self.tissue_paths[tissue_name] = this_tissue_fpath

        self.all_tissues: Set[str] = set(self.tissue_paths.keys())
        self.load_from_files()

    def load_from_files(self):
        for tissue, fpath in self.tissue_paths.items():
            logger.info(f"Loading GTEX eQTL results for {tissue} from {fpath}...")

            fpath_splat = str(fpath).split(".")
            if fpath_splat[-1] == "txt" or (
                len(fpath_splat) > 1 and fpath_splat[-2] == "txt"
            ):
                self.eqtl_data[tissue] = pd.read_csv(fpath, sep="\t")
            else:
                self.eqtl_data[tissue] = pd.read_parquet(fpath)

            logger.info(f"Annotating eqtl data for {tissue}...")
            self._annotate_eqtls(tissue)

            logger.info(f"Indexing variants and genes for {tissue}...")

            if tissue not in self._row_ids_by_variant:
                self._row_ids_by_variant[tissue] = {}
            if tissue not in self._row_ids_by_gene:
                self._row_ids_by_gene[tissue] = {}
            if tissue not in self._chroms_by_gene:
                self._chroms_by_gene[tissue] = {}
            if tissue not in self.variants_by_gene:
                self.variants_by_gene[tissue] = {}

            for row_id, spdi, gene, chrom in iterate_cols(
                self.eqtl_data[tissue],
                ("spdi", "gene", "chrom"),
                preface_with_index=True,
                tqdm_desc="Row",
            ):
                if spdi not in self._row_ids_by_variant[tissue]:
                    self._row_ids_by_variant[tissue][spdi] = []
                self._row_ids_by_variant[tissue][spdi].append(row_id)

                if gene not in self._row_ids_by_gene[tissue]:
                    self._row_ids_by_gene[tissue][gene] = []
                self._row_ids_by_gene[tissue][gene].append(row_id)

                if gene not in self.variants_by_gene[tissue]:
                    self.variants_by_gene[tissue][gene] = set([])
                self.variants_by_gene[tissue][gene].add(spdi)
                self.all_genes.add(gene)

                if gene not in self._chroms_by_gene[tissue]:
                    self._chroms_by_gene[tissue][gene] = set([])
                self._chroms_by_gene[tissue][gene].add(chrom)

                self.all_spdis.add(spdi)

    def _annotate_eqtls(self, tissue):
        self.eqtl_data[tissue]["spdi"] = [
            convert_gtex_variant_id_to_spdi(variant_id, pos_offset=-1)
            for variant_id in self.eqtl_data[tissue].variant_id
        ]
        chroms = []
        pos_s = []
        for spdi in self.eqtl_data[tissue].spdi:
            splat = spdi.split(":")
            chroms.append(splat[0])
            pos_s.append(int(splat[1]))

        self.eqtl_data[tissue]["chrom"] = chroms
        self.eqtl_data[tissue]["pos"] = pos_s

        if (
            "gene_id" in self.eqtl_data[tissue].columns
        ):  # v8 used phenotype_id, v10 uses gene_id
            self.eqtl_data[tissue]["gene"] = [
                phenotype.split(".")[0] for phenotype in self.eqtl_data[tissue].gene_id
            ]
        else:
            self.eqtl_data[tissue]["gene"] = [
                phenotype.split(".")[0]
                for phenotype in self.eqtl_data[tissue].phenotype_id
            ]

        if "maf" in self.eqtl_data[tissue].columns:
            maf_column = "maf"
        else:
            maf_column = "af"

        self.eqtl_data[tissue]["het_only"] = (
            self.eqtl_data[tissue]["ma_samples"] - self.eqtl_data[tissue]["ma_count"]
            <= 0
        )
        self.eqtl_data[tissue]["inferred_total_samples"] = (
            self.eqtl_data[tissue]["ma_count"] // self.eqtl_data[tissue][maf_column]
        )
        self.eqtl_data[tissue]["hw_hom_freq"] = self.eqtl_data[tissue][maf_column] ** 2
        self.eqtl_data[tissue]["homozygote_count"] = np.maximum(
            self.eqtl_data[tissue]["ma_count"] - self.eqtl_data[tissue]["ma_samples"], 0
        )
        self.eqtl_data[tissue]["hw_expected_hom_count"] = (
            self.eqtl_data[tissue]["hw_hom_freq"]
            * self.eqtl_data[tissue]["inferred_total_samples"]
        )
        self.eqtl_data[tissue]["hw_pval"] = scipy.stats.binom.pmf(
            k=self.eqtl_data[tissue]["homozygote_count"],
            n=self.eqtl_data[tissue]["inferred_total_samples"],
            p=self.eqtl_data[tissue]["hw_hom_freq"],
        )

        if self.spdi_translator is not None:
            self.eqtl_data[tissue]["rsid"] = self.spdi_translator.translate_spdis_to_rsids(
                spdis=self.eqtl_data[tissue]["spdi"],
                as_set=False,                
            )

    def _get_exposure_data(self, tissue, row_ids):
        return (
            self.eqtl_data[tissue]
            .loc[row_ids, ["spdi", "slope", "slope_se"]]
            .rename(columns={"slope": "exposure_beta", "slope_se": "exposure_beta_se"})
        )

    def get_exposure_data_by_variants_gene(self, tissue, variants, gene):
        gene = gene.split(".")[0]
        if gene not in self._row_ids_by_gene[tissue]:
            return self._get_empty_df(tissue)

        query_rows = set([])
        for spdi in variants:
            if spdi in self._row_ids_by_variant[tissue]:
                query_rows.update(self._row_ids_by_variant[tissue][spdi])
        query_rows.intersection_update(self._row_ids_by_gene[tissue][gene])

        return self._get_exposure_data(tissue, sorted(query_rows))

    def get_exposure_data_by_gene(self, tissue, gene):
        gene = gene.split(".")[0]
        if gene not in self._row_ids_by_gene[tissue]:
            return self._get_empty_df(tissue)

        query_rows = self._row_ids_by_gene[tissue][gene]
        return self._get_exposure_data(tissue, sorted(query_rows))

    def get_raw_exposure_data_by_gene(self, tissue, gene):
        gene = gene.split(".")[0]
        if gene not in self._row_ids_by_gene[tissue]:
            return self._get_empty_df(tissue)

        query_rows = self._row_ids_by_gene[tissue][gene]
        return self.eqtl_data[tissue].loc[query_rows]

    def _get_empty_df(self, tissue):
        return self.eqtl_data[tissue].iloc[0:0]

    def get_valid_chroms_by_gene(self, tissue, gene):
        return self._chroms_by_gene[tissue][gene]


class MetaSoftEqtls(ExposureData):
    def __init__(
        self,
        metasoft_eqtl_fpath: Path | str,
        beta_column: str = 'BETA_FE',
        beta_se_column: str = 'STD_FE',
    ) -> None:
        self.fpath = Path(metasoft_eqtl_fpath)
        self.beta_column = beta_column
        self.beta_se_column = beta_se_column

        super().__init__()

        self.metasoft_results: pd.DataFrame | None = None
        self._row_ids_by_variant: Dict[str, List[int]] = {}
        self._row_ids_by_gene: Dict[str, List[int]] = {}
        self.variants_by_gene: Dict[str, Set[str]] = {}

        self.load_from_file()

    def load_from_file(self) -> None:
        """Load and process MetaSoft results file."""
        logger.info(f'Loading metasoft results from {self.fpath} ...')
        self.metasoft_results = pd.read_csv(self.fpath, sep='\t')

        logger.info('Annotating metasoft results ...')

        metasoft_var_ids = []
        metasoft_spdis = []
        metasoft_genes = []

        for rsid in tqdm(self.metasoft_results.RSID, desc='Variants'):
            splat = rsid.split(',')
            metasoft_var_ids.append(splat[0])
            metasoft_spdis.append(convert_gtex_variant_id_to_spdi(splat[0]))
            metasoft_genes.append(splat[1].split('.')[0])

        self.metasoft_results['var_id'] = metasoft_var_ids
        self.metasoft_results['spdi'] = metasoft_spdis
        self.metasoft_results['gene'] = metasoft_genes

        logger.info('Indexing variants and genes ...')

        for row_id, spdi, gene in iterate_cols(
            self.metasoft_results,
            ('spdi', 'gene'),
            preface_with_index=True,
            tqdm_desc='Row',
        ):
            if spdi not in self._row_ids_by_variant:
                self._row_ids_by_variant[spdi] = []
            self._row_ids_by_variant[spdi].append(row_id)

            if gene not in self._row_ids_by_gene:
                self._row_ids_by_gene[gene] = []
            self._row_ids_by_gene[gene].append(row_id)

            if gene not in self.variants_by_gene:
                self.variants_by_gene[gene] = set()
            self.variants_by_gene[gene].add(spdi)

    def _get_exposure_data(self, row_ids: List[int]) -> pd.DataFrame:
        """Get exposure data for specified row IDs."""
        if self.metasoft_results is None:
            raise RuntimeError('Metasoft results not loaded')
            
        return self.metasoft_results.loc[
            row_ids, ['spdi', self.beta_column, self.beta_se_column]
        ].rename(
            columns={
                self.beta_column: 'exposure_beta',
                self.beta_se_column: 'exposure_beta_se',
            }
        )

    def get_exposure_data_by_variants_gene(
        self, 
        variants: List[str], 
        gene: str
    ) -> pd.DataFrame:
        """Get exposure data for variants and gene."""
        gene = gene.split('.')[0]
        query_rows = set()
        if gene not in self._row_ids_by_gene:
            return self._get_empty_df()

        for spdi in variants:
            if spdi in self._row_ids_by_variant:
                query_rows.update(self._row_ids_by_variant[spdi])
        query_rows.intersection_update(self._row_ids_by_gene[gene])

        return self._get_exposure_data(sorted(query_rows))

    def get_exposure_data_by_gene(self, gene: str) -> pd.DataFrame:
        """Get all exposure data for a gene."""
        gene = gene.split('.')[0]
        if gene not in self._row_ids_by_gene:
            return self._get_empty_df()

        query_rows = self._row_ids_by_gene[gene]
        return self._get_exposure_data(sorted(query_rows))

    def _get_empty_df(self) -> pd.DataFrame:
        """Return empty DataFrame with correct columns."""
        if self.metasoft_results is None:
            raise RuntimeError('Metasoft results not loaded')
        return self.metasoft_results.iloc[0:0]

    @property
    def all_genes(self) -> Set[str]:
        """Get set of all genes."""
        return set(self._row_ids_by_gene.keys())


class OpentargetsResults:
    BETA_COL_PAIRS = [("beta", "beta_se"), ("log_oddsr", "log_oddsr_se")]

    def __init__(
        self,
        lead_variants_only: bool = False,
        drop_missing_beta_rows: bool = True,
        trait_column: str = DEFAULT_TRAIT_COLUMN,
    ):
        self.lead_variants_only = lead_variants_only
        self.drop_missing_beta_rows = drop_missing_beta_rows
        self.trait_column = trait_column

        self._row_ids_by_variant = {}
        self._row_ids_by_study = {}
        self._row_ids_by_chrom = {}
        self._row_ids_by_trait = {}
        self.variants_by_study = {}
        self.variants_by_trait = {}
        self.studies_by_trait = {}
        self.ot_results = None

    @property
    def all_spdis(self):
        return set(self._row_ids_by_variant.keys())
    
    @property
    def all_traits(self):
        return set(self._row_ids_by_trait.keys())

    @classmethod
    def from_file(
        cls,
        file: io.TextIOWrapper | Path | str,
        file_format="tsv",
        lead_variants_only: bool = False,
        drop_missing_beta_rows: bool = True,
        trait_column: str = DEFAULT_TRAIT_COLUMN,
    ) -> Self:
        logger.info(f"Loading OpenTargets results from {file} ...")

        if file_format == "parquet":
            ot_results = pd.read_parquet(file)
        elif file_format == "tsv":
            ot_results = pd.read_csv(
                file,
                sep="\t",
            )
        else:
            ot_results = pd.read_csv(file)

        return cls.from_dataframe(
            ot_results,
            lead_variants_only=lead_variants_only,
            drop_missing_beta_rows=drop_missing_beta_rows,
            trait_column=trait_column,
        )

    @classmethod
    def from_dataframe(
        cls,
        result_dataframe: pd.DataFrame,
        spdi_translator=None,
        lead_variants_only: bool = False,
        drop_missing_beta_rows: bool = True,
        trait_column: str = DEFAULT_TRAIT_COLUMN,
    ) -> Self:
        new_obj = cls(
            drop_missing_beta_rows=drop_missing_beta_rows,
            trait_column=trait_column,
        )
        new_obj.ot_results = result_dataframe

        unfiltered_row_count = new_obj.ot_results.shape[0]
        logger.info(f"Loaded {unfiltered_row_count} rows.")

        if new_obj.drop_missing_beta_rows:
            new_obj.ot_results = new_obj.ot_results.dropna(
                how="all", subset=["beta", "odds_ratio"]
            ).copy()
            filtered_row_count = new_obj.ot_results.shape[0]
            logger.info(
                f"Removed rows with empty beta values. {filtered_row_count} rows remain."
            )

        if lead_variants_only:
            new_obj.ot_results = new_obj.ot_results.drop_duplicates(
                subset=("lead_spdi", "trait_reported", "beta", "odds_ratio", "study_id")
            )

        logger.info("Annotating ...")

        new_obj.ot_results["beta_se"] = (
            new_obj.ot_results["beta_ci_upper"] - new_obj.ot_results["beta_ci_lower"]
        ) / Z_95
        new_obj.ot_results["log_oddsr"] = np.log(new_obj.ot_results["odds_ratio"])
        new_obj.ot_results["log_oddsr_se"] = (
            np.log(new_obj.ot_results["oddsr_ci_upper"])
            - np.log(new_obj.ot_results["oddsr_ci_lower"])
        ) / Z_95

        if spdi_translator is not None:
            new_obj.ot_results["lead_rsid"] = spdi_translator.translate_spdis_to_rsids(
                new_obj.ot_results["lead_spdi"]
            )
            new_obj.ot_results["tag_rsid"] = spdi_translator.translate_spdis_to_rsids(
                new_obj.ot_results["tag_spdi"]
            )

        fixed_traits = []
        trait_translations = {}

        for trait in new_obj.ot_results[new_obj.trait_column]:
            if trait not in trait_translations:
                trait_translations[trait] = new_obj.fix_trait_name(trait)
            fixed_traits.append(trait_translations[trait])
        new_obj.ot_results[new_obj.trait_column] = fixed_traits

        logger.info("Indexing studies and variants ...")

        for row_idx, spdi, trait, study, chrom in iterate_cols(
            new_obj.ot_results,
            ("lead_spdi", new_obj.trait_column, "study_id", "chrom"),
            preface_with_index=True,
        ):
            if spdi not in new_obj._row_ids_by_variant:
                new_obj._row_ids_by_variant[spdi] = []
            new_obj._row_ids_by_variant[spdi].append(row_idx)

            if study not in new_obj._row_ids_by_study:
                new_obj._row_ids_by_study[study] = []
            new_obj._row_ids_by_study[study].append(row_idx)

            if study not in new_obj.variants_by_study:
                new_obj.variants_by_study[study] = set([])
            new_obj.variants_by_study[study].add(spdi)

            if chrom not in new_obj._row_ids_by_chrom:
                new_obj._row_ids_by_chrom[chrom] = []
            new_obj._row_ids_by_chrom[chrom].append(row_idx)

            if trait not in new_obj._row_ids_by_trait:
                new_obj._row_ids_by_trait[trait] = []
                new_obj.variants_by_trait[trait] = set([])
                new_obj.studies_by_trait[trait] = set([])
            new_obj._row_ids_by_trait[trait].append(row_idx)
            new_obj.variants_by_trait[trait].add(spdi)
            new_obj.studies_by_trait[trait].add(study)

        new_obj._all_studies = set(new_obj._row_ids_by_study.keys())
        new_obj._all_spdis = set(new_obj._row_ids_by_variant.keys())
        new_obj._all_traits = set(new_obj._row_ids_by_trait.keys())
        logger.info("Done loading Opentargets information.")

        return new_obj

    def _get_outcome_data(self, row_ids, auto_collapse_beta_columns=True):
        row_subset = self.ot_results.loc[row_ids]

        max_non_nans = 0
        best_beta_cols = self.BETA_COL_PAIRS[0]
        if auto_collapse_beta_columns:
            for beta_col, beta_se_col in self.BETA_COL_PAIRS:
                num_non_nans = np.logical_not(np.isnan(row_subset[beta_col])).sum()

                if num_non_nans > max_non_nans:
                    max_non_nans = num_non_nans
                    best_beta_cols = (beta_col, beta_se_col)
            row_subset["beta_col"] = beta_col
            return (
                row_subset.loc[
                    :,
                    [
                        "lead_spdi",
                        best_beta_cols[0],
                        best_beta_cols[1],
                        "beta_col",
                        "study_id",
                    ],
                ]
                .dropna(subset=best_beta_cols, how="any")
                .rename(
                    columns={
                        "lead_spdi": "spdi",
                        best_beta_cols[0]: "outcome_beta",
                        best_beta_cols[1]: "outcome_beta_se",
                        "study_id": "study_id",
                    }
                )
                .drop_duplicates()
            )

        return (
            row_subset.loc[
                :,
                [
                    "lead_spdi",
                    "beta",
                    "beta_se",
                    "log_oddsr",
                    "log_oddsr_se",
                    "study_id",
                ],
            ]
            .rename(
                columns={
                    "lead_spdi": "spdi",
                    "beta": "outcome_beta",
                    "beta_se": "outcome_beta_se",
                    "log_oddsr": "outcome_log_oddsr",
                    "log_oddsr_se": "outcome_log_oddsr_se",
                    "study_id": "study_id",
                }
            )
            .drop_duplicates()
        )

    def get_outcome_data_by_variants(self, variants, auto_collapse_beta_columns=True):
        query_rows = set([])

        for spdi in variants:
            if spdi in self._row_ids_by_variant:
                query_rows.update(self._row_ids_by_variant[spdi])

        return self._get_outcome_data(
            sorted(query_rows), auto_collapse_beta_columns=auto_collapse_beta_columns
        )

    def get_outcome_data_by_study(self, study_id: str, auto_collapse_beta_columns=True):
        return self._get_outcome_data(
            sorted(self._row_ids_by_study[study_id]),
            auto_collapse_beta_columns=auto_collapse_beta_columns,
        )

    def get_outcome_data_by_variants_study(
        self, variants: Iterable[str], study_id: str, auto_collapse_beta_columns=True
    ):
        query_rows = set([])

        for spdi in variants:
            if spdi in self._row_ids_by_variant:
                query_rows.update(self._row_ids_by_variant[spdi])

        query_rows.intersection_update(self._row_ids_by_study[study_id])

        return self._get_outcome_data(sorted(query_rows))

    def get_outcome_data_by_variants_study_chroms(
        self,
        variants: Iterable[str],
        study_id: str,
        chromosomes: Iterable[str],
        auto_collapse_beta_columns=True,
    ):
        query_rows = set([])

        for spdi in variants:
            if spdi in self._row_ids_by_variant:
                query_rows.update(self._row_ids_by_variant[spdi])

        query_rows.intersection_update(self._row_ids_by_study[study_id])
        valid_row_ids_by_chrom = set([])
        for chrom in chromosomes:
            valid_row_ids_by_chrom.update(self._row_ids_by_chrom[chrom])
        query_rows.intersection_update(valid_row_ids_by_chrom)

        return self._get_outcome_data(sorted(query_rows))

    def get_empty_df(self):
        return self.ot_results.iloc[0:0]

    @staticmethod
    def fix_trait_name(trait_name):
        return trait_name.strip('"!')

    @property
    def all_studies(self):
        return self._all_studies

    @classmethod
    def load_opentargets_by_trait(
        cls,
        opentargets_results_fpath: Path | str,
        spdi_translator=None,
        lead_variants_only: bool = False,
        drop_missing_beta_rows: bool = True,
        trait_column: str = DEFAULT_TRAIT_COLUMN,
        drop_empty_traits: bool = True,
        verbose: bool = False,
    ) -> Self:
        opentargets_results_fpath = Path(opentargets_results_fpath)
        print(f"Loading Opentargets results from {opentargets_results_fpath} ...")
        if opentargets_results_fpath.suffix == ".parq":
            full_ot_results = pd.read_parquet(opentargets_results_fpath)
        else:
            full_ot_results = pd.read_csv(opentargets_results_fpath, sep="\t")

        if spdi_translator is not None:
            full_ot_results["lead_rsid"] = spdi_translator.translate_spdis_to_rsids(
                full_ot_results["lead_spdi"], as_set=False
            )
            full_ot_results["tag_rsid"] = spdi_translator.translate_spdis_to_rsids(
                full_ot_results["tag_spdi"], as_set=False
            )

        print("Partitioning Opentargets results by trait ...")

        ot_results_by_trait = {}
        for trait, trait_results in tqdm(
            full_ot_results.groupby(trait_column), desc="Trait"
        ):
            trait = OpentargetsResults.fix_trait_name(trait)
            if verbose:
                print(
                    f"Processing trait {trait} with {trait_results.shape[0]} rows ..."
                )

            this_trait_ot_results = OpentargetsResults.from_dataframe(
                trait_results,
                spdi_translator=None,
                lead_variants_only=lead_variants_only,
                drop_missing_beta_rows=drop_missing_beta_rows,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"{this_trait_ot_results.ot_results.shape[0]} rows remain after processing for trait {trait}."
                )

            if this_trait_ot_results.ot_results.shape[0] > 0 or not drop_empty_traits:
                ot_results_by_trait[trait] = this_trait_ot_results

        print(
            f"Generated Opentargets results for {len(ot_results_by_trait)} traits ..."
        )

        return ot_results_by_trait


class MRData:
    def __init__(
        self,
        exposure_data: GtexEqtls,
        outcome_data: OpentargetsResults,
        spdi_translator: SpdiRsidTranslatorDbSnp,
        variant_corrs: VariantCorrelations,
        variant_proxies: VariantProxies,
    ) -> None:
        self.exposure_data = exposure_data
        self.outcome_data = outcome_data
        self.spdi_translator = spdi_translator
        self.variant_correlations = variant_corrs
        self.variant_proxies = variant_proxies

        self.variants_by_tissue_trait_gene: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
        self.meta_tissue_genes_checked: Set[str] = set()
        self._exposure_spdis: Set[str] = self.exposure_data.all_spdis
        self._outcome_spdis: Set[str] = self.outcome_data.all_spdis
        self._all_spdis: Set[str] = self._exposure_spdis.union(self._outcome_spdis)

        logger.info('Loading SPDI/RSID translations for all variants...')
        self.spdi_to_rsid: Dict[str, str]
        self.rsid_to_spdi: Dict[str, str]
        self.spdi_to_rsid, self.rsid_to_spdi = (
            self.spdi_translator.get_translations_bidirectional(self.all_spdis)
        )

        logger.info('Getting LD proxies for all variants...')
        self._all_rsids: Set[str] = {
            self.spdi_to_rsid[spdi] 
            for spdi in self._all_spdis 
            if spdi in self.spdi_to_rsid
        }
        self.proxies_by_rsid: Dict[str, Set[str]] = variant_proxies.get_variant_proxies(self._all_rsids)

    @property
    def all_spdis(self):
        """
        Returns a set of all SPDIs across all exposure and outcome datasets
        """
        return self._all_spdis

    @property
    def all_rsids(self):
        """
        Returns a set of all RSIDs across all exposure and outcome datasets
        """
        return self._all_rsids

    def _generate_spdi_rsid_translations(self):
        self.spdi_to_rsid, self.rsid_to_spdi = (
            self.spdi_translator.get_translations_bidirectional(self.all_spdis)
        )

    @property
    def exposure_spdis(self):
        return self._exposure_spdis

    @property
    def outcome_spdis(self):
        return self._outcome_spdis

    def _match_variants_by_proxy(
        self, exp_spdis, out_spdis, min_proxy_r2: float = None
    ):
        exp_spdis = set(exp_spdis)
        out_spdis = set(out_spdis)

        logger.debug(f"{len(exp_spdis)} exp_spdis")
        logger.debug(f"{len(out_spdis)} out_spdis")
        # # Make pairs out of trivially self-overlapping variants, and assign the rest to two queues for further processing
        overlapping_spdis = out_spdis.intersection(exp_spdis)
        exp_spdis.difference_update(overlapping_spdis)
        out_spdis.difference_update(overlapping_spdis)
        exp_out_spdi_pairs = set([(spdi, spdi) for spdi in overlapping_spdis])
        logger.debug(f"{len(overlapping_spdis)} overlapping_spdis")
        logger.debug(f"{len(exp_spdis)} non-overlapping exp_spdis")
        logger.debug(f"{len(out_spdis)} non-overlapping out_spdis")

        if (
            len(out_spdis) == 0 or len(exp_spdis) == 0
        ):  # If there's nothing left to match, stop early
            return exp_out_spdi_pairs

        # Get spdi-to-rsid and rsid-to-spdi dicts so we can use rsids with plink and convert back afterward
        # this_locus_spdi_to_rsid, this_locus_rsid_to_spdi = self.spdi_translator.get_translations_bidirectional(exp_spdis.union(out_spdis))

        # Construct a dict with a subset of the proxy graph restricted to the exposure variants in the sources and outcome variants in the targets. Keep track of the variants so encompassed.
        exp_rsids = set([])

        for exp_spdi in exp_spdis:
            if exp_spdi in self.spdi_to_rsid:
                exp_rsids.add(self.spdi_to_rsid[exp_spdi])

        out_rsids = set([])
        proxy_rsids = set([])
        candidate_proxy_graph_dict = {}

        for out_spdi in out_spdis:
            if out_spdi in self.spdi_to_rsid:
                out_rsid = self.spdi_to_rsid[out_spdi]
                logger.debug(
                    f"Searching for proxies for outcome variant {out_spdi} {out_rsid} ..."
                )
                if out_rsid in self.proxies_by_rsid and len(
                    self.proxies_by_rsid[out_rsid]
                ):
                    logger.debug(
                        f"{out_rsid} has {len(self.proxies_by_rsid[out_rsid])} proxies"
                    )
                    candidate_exposure_proxies = exp_rsids.intersection(
                        self.proxies_by_rsid[out_rsid]
                    )
                    logger.debug(
                        f"The following proxies are also exposure variants: {candidate_exposure_proxies}"
                    )
                    # print(self.variant_proxies.proxies[exp_rsid])
                    # print(f'Has candidate proxies {candidate_outcome_proxies}')
                    if len(candidate_exposure_proxies):
                        out_rsids.add(out_rsid)
                        proxy_rsids.update(candidate_exposure_proxies)
                        candidate_proxy_graph_dict[out_rsid] = (
                            candidate_exposure_proxies
                        )
        if (
            len(out_rsids) == 0 or len(proxy_rsids) == 0
        ):  # If there's nothing left to match, stop early
            return exp_out_spdi_pairs

        # Now get the correlation matrix for all exposure and candidate proxy variants
        proxy_corrs = (
            self.variant_correlations.get_matrix_rsids(out_rsids.union(proxy_rsids))
            .sort_index()
            .sort_index(axis=1)
        )
        all_rsids_with_corrs = set(proxy_corrs.index)
        out_rsids.intersection_update(all_rsids_with_corrs)
        proxy_rsids.intersection_update(all_rsids_with_corrs)

        # Enumerate the "edges" and sort by (R2) weights in descending order
        edge_q = []
        for out_rsid in out_rsids:
            for candidate_proxy_rsid in candidate_proxy_graph_dict[out_rsid]:
                weight = proxy_corrs.loc[out_rsid, candidate_proxy_rsid]
                if min_proxy_r2 is None or weight >= min_proxy_r2:
                    edge_q.append((weight, out_rsid, candidate_proxy_rsid))
        edge_q.sort(key=lambda x: x[0], reverse=True)

        # Iterate through the edges, taking the strongest weighted ones as matched pairs, removing their nodes from the appropriate queues as we go.
        for weight, out_rsid, candidate_proxy_rsid in edge_q:
            if out_rsid not in out_rsids or candidate_proxy_rsid not in exp_rsids:
                continue

            out_spdi = self.rsid_to_spdi[out_rsid]
            candidate_proxy_spdi = self.rsid_to_spdi[candidate_proxy_rsid]
            exp_out_spdi_pairs.add((candidate_proxy_spdi, out_spdi))
            out_rsids.remove(out_rsid)
            exp_rsids.remove(candidate_proxy_rsid)

            if not len(exp_rsids) or not len(out_rsids):
                break

        return exp_out_spdi_pairs

    def get_model_data_uncorrelated(self, tissue: str, trait: str, gene: str) -> Dict[str, pd.DataFrame]:
        """Get uncorrelated model data for a tissue-trait-gene combination.
        
        Args:
            tissue: GTEx tissue name
            trait: Trait name from outcome data
            gene: Gene name/ID (will be trimmed to ENSG format)
            
        Returns:
            Dictionary mapping study IDs to DataFrames containing model data
            
        Raises:
            KeyError: If trait or tissue not found
        """
        if trait not in self.outcome_data.all_traits:
            raise KeyError(f'Trait \'{trait}\' not found!')
        if tissue not in self.exposure_data.all_tissues:
            raise KeyError(f'Tissue \'{tissue}\' not found!')

        gene = trim_ensembl_gene(gene)
        if gene not in self.exposure_data.variants_by_gene[tissue]:
            logger.info(f'No variants found for gene {gene} in tissue {tissue}')
            return {}

        tissue_gene_exposure_spdis = self.exposure_data.variants_by_gene[tissue][gene]
        valid_chroms = self.exposure_data.get_valid_chroms_by_gene(tissue, gene)

        model_data_by_study: Dict[str, pd.DataFrame] = {}

        for study in sorted(self.outcome_data.all_studies):
            logger.debug(f'Analyzing study {study} ...')

            trait_study_outcome_spdis = [
                spdi 
                for spdi in self.outcome_data.variants_by_study[study]
                if spdi.split(':')[0] in valid_chroms
            ]
            if not trait_study_outcome_spdis:
                logger.debug('No valid outcome study variants remain after chromosome filtering')
                continue

            matched_variant_pairs = self._match_variants_by_proxy(
                tissue_gene_exposure_spdis, 
                trait_study_outcome_spdis
            )

            if not matched_variant_pairs:
                logger.debug('No matched variants!')
                continue

            model_df_data = []

            for exposure_spdi, outcome_spdi in matched_variant_pairs:
                try:
                    exposure_row = self.exposure_data.get_exposure_data_by_variants_gene(
                        tissue=tissue,
                        variants=[exposure_spdi], 
                        gene=gene
                    )
                    if exposure_row.shape[0] != 1:
                        logger.warning(f'Expected 1 exposure row for {exposure_spdi}, got {exposure_row.shape[0]}')
                        continue

                    outcome_row = self.outcome_data.get_outcome_data_by_variants_study(
                        variants=[outcome_spdi], 
                        study_id=study
                    )
                    if outcome_row.shape[0] != 1:
                        logger.warning(f'Expected 1 outcome row for {outcome_spdi}, got {outcome_row.shape[0]}')
                        continue

                    model_df_data.append({
                        'spdi': exposure_spdi,
                        'exposure_beta': exposure_row.exposure_beta.iloc[0],
                        'exposure_beta_se': exposure_row.exposure_beta_se.iloc[0], 
                        'outcome_beta': outcome_row.outcome_beta.iloc[0],
                        'outcome_beta_se': outcome_row.outcome_beta_se.iloc[0]
                    })
                except Exception as e:
                    logger.error(f'Error processing variant pair {exposure_spdi}, {outcome_spdi}: {str(e)}')
                    continue
                    
            if model_df_data:
                model_data_by_study[study] = pd.DataFrame(model_df_data)

        return model_data_by_study

    @staticmethod
    def align_betas_rho(betas_df, rho_df):
        # Truncate betas to only variants with hits in LD-LINK
        betas_df = betas_df.loc[my_in1d(betas_df.spdi, rho_df.index)]
        spdi_index = list(betas_df.spdi)

        # Expand rho to reflect all entries in betas_df (which can have multiple rows per SPDI if combining studies)
        expanded_rho = pd.DataFrame(index=spdi_index, columns=spdi_index)
        for row_num, row_spdi in enumerate(spdi_index):
            for col_num, col_spdi in enumerate(spdi_index):
                expanded_rho.iloc[row_num, col_num] = rho_df.loc[row_spdi, col_spdi]

        return betas_df.sort_values("spdi"), expanded_rho.astype(float).sort_index(
            axis=0
        ).sort_index(axis=1)

    def get_model_data_correlated(self, tissue, trait, gene, combine_studies=False):
        gene = trim_ensembl_gene(gene)
        # print(tissue, trait, gene)
        uncorrelated_model_data = self.get_model_data_uncorrelated(
            tissue=tissue, trait=trait, gene=gene
        )
        # print(uncorrelated_model_data)

        correlated_model_data = {}
        for study, betas_df in uncorrelated_model_data.items():
            if betas_df.shape[0] == 0:
                logger.info(f"\tNo variants found in study {study}, skipping ...")
                continue
            logger.info(
                f"\tGetting correlations for {betas_df.shape[0]} variants in study {study} ..."
            )
            corrs_df = self.variant_correlations.get_matrix(betas_df.spdi)
            logger.info(f"\tFound correlations for {corrs_df.shape[0]} variants.")
            betas_df, rho_df = self.align_betas_rho(betas_df, corrs_df)

            correlated_model_data[study] = (betas_df, rho_df)

        return correlated_model_data

    def pre_populate_correlations(self, gene, verbose=False):
        if gene not in self.meta_tissue_genes_checked:
            logger.info(
                f"Pre-populating correlations for all eQTLs across tissues for {gene} ..."
            )

            this_gene_all_tissue_variants = self.get_all_genic_variants(gene)
            _ = self.variant_correlations.get_matrix(
                this_gene_all_tissue_variants
            )
            self.meta_tissue_genes_checked.add(gene)

    def get_all_genic_variants(self, gene, filter_by_outcome: bool = False):
        this_gene_all_tissue_variants = set([])
        for this_tissue in self.exposure_data.all_tissues:
            if gene in self.exposure_data.variants_by_gene[this_tissue]:
                this_gene_all_tissue_variants.update(
                    self.exposure_data.variants_by_gene[this_tissue][gene]
                )

        # ToDo: make wrapper classes for the outcome and exposure dictionaries
        if filter_by_outcome:
            this_gene_all_tissue_variants.intersection_update(self.outcome_spdis)

        return this_gene_all_tissue_variants


# Alternative method of loading opentargets data. Not pursued further since seems no faster than doing a groupby()
def load_opentargets_by_trait_csv(
    opentargets_results_fpath: Path | str,
    drop_missing_beta_rows: bool = True,
    trait_column: str = DEFAULT_TRAIT_COLUMN,
    verbose: bool = True,
) -> Dict[str, OpentargetsResults]:
    opentargets_results_fpath = Path(opentargets_results_fpath)
    if verbose:
        print(f"Loading Opentargets results from {opentargets_results_fpath} ...")

    if opentargets_results_fpath.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    ot_results_by_trait = collections.defaultdict(lambda: [])
    first_line = True
    # line_cnt = 0
    with opener(opentargets_results_fpath, "rt") as in_file:
        if first_line:
            first_line = False
            col_positions = {}
            columns = in_file.readline().strip().split("\t")

            for i, col_name in enumerate(columns):
                col_positions[col_name] = i
            trait_col_pos = col_positions[trait_column]

        for line in tqdm(in_file, desc="Lines"):
            splat = line.strip().split("\t")
            trait = OpentargetsResults.fix_trait_name(splat[trait_col_pos])
            ot_results_by_trait[trait].append(splat)

    for trait in tqdm(ot_results_by_trait, desc="Traits"):
        ot_results_by_trait[trait] = pd.DataFrame(
            ot_results_by_trait[trait], columns=columns
        )
        convert_dtypes_correctly(ot_results_by_trait[trait])
        # print(ot_results_by_trait[trait].dtypes)
        # return(ot_results_by_trait[trait])
        ot_results_by_trait[trait] = OpentargetsResults.from_dataframe(
            ot_results_by_trait[trait],
            drop_missing_beta_rows=drop_missing_beta_rows,
            verbose=False,
        )

    if verbose:
        print(
            f"Generated Opentargets results for {len(ot_results_by_trait)} traits ..."
        )

    return ot_results_by_trait