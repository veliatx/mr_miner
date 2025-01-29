import gzip
import logging
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from mr_miner._version import version as __version__
except ImportError:
    __version__ = "unknown"  # Fallback when _version.py hasn't been generated yet

from mr_miner import data_wrappers, mr_models
from mr_miner.constants import (
    DEFAULT_POP,
    DEFAULT_THREADS,
    DEFAULT_PROXY_R2_THRESHOLD,
    DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
    DEFAULT_OUTCOME_DATA_PATH,
    MIN_DET,
)
from mr_miner.plink_wrapper import PlinkWrapper
from mr_miner.variant_correlations import VariantCorrelations
from mr_miner.variant_proxies import VariantProxies
from mr_miner.config import MRConfig
from mr_miner.chrom_mapper import ChromMapper
from mr_miner.variant_ids import (
    initialize_spdi_translator
)
# Import and configure logger
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up and execute the argument parser"""
    parser = argparse.ArgumentParser(
        description='Assess the causal relationship between a set of exposures '
        'and outcomes using public summary statistics.'
    )

    # Required arguments
    parser.add_argument(
        '-o', '--output_prefix',
        type=str,
        required=True,
        help='The prefix for the output files'
    )
    parser.add_argument(
        '-t', '--thousand_genomes_basepath',
        type=str,
        required=True,
        help='The basepath for the thousand genomes data'
    )
    parser.add_argument(
        '-e', '--ensembl_gff',
        type=str,
        required=True,
        help='Path to Ensembl GFF annotation file'
    )
    parser.add_argument(
        '-c', '--cache_path',
        type=str,
        required=True,
        help='Path to store cache files'
    )
    parser.add_argument(
        '--eqtl_data_path',
        type=str,
        required=True,
        help='Path to directory containing GTEx eQTL data files'
    )
    parser.add_argument(
        '--outcome_data_path',
        type=str,
        required=False,
        default=DEFAULT_OUTCOME_DATA_PATH,
        help='Path to directory containing OpenTargets variant info data'
    )

    # Variant ID translation source group
    variant_id_group = parser.add_mutually_exclusive_group(required=True)
    variant_id_group.add_argument(
        '--dbsnp_vcf_fpath',
        type=str,
        help='dbSNP VCF file'
    )
    variant_id_group.add_argument(
        '--use_1kg_variants',
        action='store_true',
        help='Use 1000 Genomes variants for SPDI-rsID translation'
    )

    # Add gene list arguments in a mutually exclusive group
    gene_group = parser.add_mutually_exclusive_group()
    gene_group.add_argument(
        '--genes',
        type=str,
        help='Comma-separated list of gene names to analyze'
    )
    gene_group.add_argument(
        '--gene-file',
        type=Path,
        help='Path to file containing gene names, one per line'
    )

    # Optional arguments
    parser.add_argument(
        '-p', '--pop',
        type=str,
        default=DEFAULT_POP,
        help=f'Population code for LD calculations (default: {DEFAULT_POP})'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Whether to run in verbose mode (show DEBUG messages)'
    )
    parser.add_argument(
        '-f', '--config_file',
        type=str,
        help='Path to configuration file (overrides other arguments)'
    )
    parser.add_argument(
        '--plink_fpath',
        type=str,
        default='plink',
        help='Path to plink executable'
    )
    parser.add_argument(
        '--proxy_r2_threshold', 
        type=float,
        default=DEFAULT_PROXY_R2_THRESHOLD,
        help=f'R² threshold for proxy variants (default: {DEFAULT_PROXY_R2_THRESHOLD})'
    )
    parser.add_argument(
        '--proxy_distance_threshold_kb',
        type=float,
        default=DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
        help=f'Distance threshold in kb for proxy variants (default: {DEFAULT_PROXY_DISTANCE_THRESHOLD_KB})'
    )
    parser.add_argument('-@',
        '--threads',
        type=int,
        default=DEFAULT_THREADS,
        help=f'Number of threads for multiprocessing (default: {DEFAULT_THREADS})'
    )
    
    return parser.parse_args()


def setup_logger(log_fpath: Path | str, verbosity: int = logging.INFO) -> None:
    """Set up logging configuration"""
    log_fpath = Path(log_fpath)
    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(verbosity)
    
    # File handler - changed to 'w' mode
    fh = logging.FileHandler(log_fpath, mode='w')
    fh.setLevel(verbosity)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(verbosity)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)


def assess_tissue_trait_gene(mr_data, query_tissue, query_trait, query_gene):
    these_results = []

    logger.info(f'Assessing {query_tissue}, {query_trait}, {query_gene} ...')

    uncorrelated_model_data_by_study = mr_data.get_model_data_uncorrelated(query_tissue, query_trait, query_gene)               

    try: 
        correlated_model_data_by_study = mr_data.get_model_data_correlated(query_tissue, query_trait, query_gene)
    except Exception as e:
        logger.error(f'\tFailed to retrieve correlated model data for {query_gene} {query_tissue} {query_trait}, {e}.skipping.')
        correlated_model_data_by_study = {}

    for query_study in set(uncorrelated_model_data_by_study.keys()).union(correlated_model_data_by_study.keys()):
        this_result = {}

        logger.info(f'Examining study {query_study} ...')

        this_result['study'] = query_study
        this_result['tissue'] = query_tissue
        this_result['trait'] = query_trait
        this_result['gene'] = query_gene        

        if query_study in uncorrelated_model_data_by_study:
            betas_df = uncorrelated_model_data_by_study[query_study]
            this_result['uncorrelated_num_loci'] = len(set(betas_df.spdi))
            this_result['uncorrelated_num_points'] = betas_df.shape[0]

            if betas_df.shape[0] == 0:
                logger.info('\tNo shared uncorrelated variants')                
                continue
        
            this_result['uncorrelated_exposure_beta_mean'] = betas_df.exposure_beta.mean()
            this_result['uncorrelated_outcome_beta_mean'] = betas_df.outcome_beta.mean()

            if betas_df.shape[0] == 1:
                logger.info('\tComputing Wald estimate')
                this_model = mr_models.WaldEstimator()
            else:
                logger.info('\tRunning uncorrelated model')                
                this_model = mr_models.BurgessUncorrelated()

            try:
                this_model.fit(betas_df)
            except (np.linalg.LinAlgError, RuntimeWarning):
                pass
            else:
                this_result.update(this_model.params)
            
        if query_study in correlated_model_data_by_study:
            correlated_betas_df, rho_df = correlated_model_data_by_study[query_study]
            this_result['correlated_num_loci'] = len(set(correlated_betas_df.spdi))
            this_result['correlated_num_points'] = correlated_betas_df.shape[0]
                    
            if correlated_betas_df.shape[0] > 0:
                this_result['correlated_exposure_beta_mean'] = correlated_betas_df.exposure_beta.mean()
                this_result['correlated_outcome_beta_mean'] = correlated_betas_df.outcome_beta.mean()
                
                # print('computing det and cond')
                this_det = np.linalg.det(rho_df) 
                this_cond = np.linalg.cond(rho_df)
                if this_det< MIN_DET:
                    logger.info(f'\tRho matrix near-singluar, determinant {this_det}, condition number {this_cond}!')
                # print('/computing det and cond')
                # break
                
                this_result['rho_det'] = this_det
                this_result['rho_cond'] = this_cond
                this_result['rho_mean'] = rho_df.mean().mean()
            
                logger.info('\tRunning principal components model')

                # print(correlated_betas_df, rho_df)

                this_model = mr_models.BurgessPCA()
                try:
                    # this_model.fit(correlated_betas_df, rho_df)
                    pass
                except (np.linalg.LinAlgError, RuntimeWarning, ValueError) as e:
                    logger.info(f'\tFailed to fit PCA model with error: {e}')
                    this_result.update({'beta_IVW_pca': np.NaN,
                        'beta_se_IVW_pca_fixed': np.NaN
                        })
                else:
                    this_result.update(this_model.params)

            else:
                logger.info('\tNo correlated variants remain')

        these_results.append(this_result)
        print(this_result)
    return these_results    


def run_mr_miner(mr_data, gene_list, output_fpath):
    ROW_BUFFER_FULL_LENGTH = 100

    row_buffer = []
    result_count = 0
    first_write = True

    logger.info(f"Running MR Miner for {len(gene_list)} genes")

    with gzip.open(output_fpath, 'wt') as results_file:
        for gene_num, query_gene in tqdm(list(enumerate(sorted(gene_list))), desc='Genes'):
            logger.info(f'Prepopulating correlations for {query_gene} ({gene_num} of {len(gene_list)}) ...')
            try:
                mr_data.pre_populate_correlations(query_gene) 
            except Exception as e:
                logger.warning(f'Failed to retrieve correlations for all eqtls in gene {query_gene} with exception {e}. Will try individual tissue-traits.')
            
            for tissue_num, query_tissue in tqdm(list(enumerate(sorted(mr_data.exposure_data.all_tissues))), desc=f'{query_gene} Tissues'):

                if query_gene not in mr_data.exposure_data.variants_by_gene[query_tissue]:
                    logger.debug(f'No eQTLS for {query_gene} in {query_tissue} ({tissue_num} of {len(mr_data.exposure_data)})')
                    continue

                this_tissue_gene_variants = set(mr_data.exposure_data.variants_by_gene[query_tissue][query_gene])
                logger.debug(f'Found {len(this_tissue_gene_variants)} variants  for {query_gene} in {query_tissue} ({tissue_num} of {len(mr_data.exposure_data.all_tissues)})')
                
                # for query_trait in tqdm(sorted(test_mr_data.outcome_data), desc=f'{query_gene} {query_tissue} Traits'):  
                for query_trait in sorted(mr_data.outcome_data.all_traits):                        
                    this_trait_variants = mr_data.outcome_data.variants_by_trait[query_trait]
                    # logger.debug(f'Found {len(this_trait_variants)} variants for trait {query_trait}')
                    
                    this_tissue_trait_gene_variants = this_tissue_gene_variants.intersection(this_trait_variants)
                    
                    if len(this_tissue_trait_gene_variants) == 0:
                        # logger.debug(f'No trait variants for trait {query_trait} in tissue {query_tissue} for gene {query_gene}.')                    
                        continue

                    logger.debug(f'Found {len(this_tissue_trait_gene_variants)} variants for trait {query_trait} in tissue {query_tissue} for gene {query_gene}.')
                    
                    try:
                        this_result_chunk = pd.DataFrame(assess_tissue_trait_gene(mr_data,
                                                                            query_tissue=query_tissue, 
                                                                            query_trait=query_trait,
                                                                            query_gene=query_gene, 
                                                                            ),
                                                    )
                    except Exception as e:
                        logger.error(f'Error analyzing {query_gene}, {query_tissue}, {query_trait}: {e}!')
                    else:
                        if this_result_chunk.shape[0] > 0:    
                            row_buffer += [this_result_chunk]
                            result_count += this_result_chunk.shape[0]
            if len(row_buffer) > ROW_BUFFER_FULL_LENGTH:
                logger.info(f'Row buffer has {len(row_buffer)} rows, writing to {output_fpath}.')                    
                pd.concat(row_buffer).to_csv(results_file, sep='\t',
                                    index=False,
                                    header=first_write)
                first_write=False
                results_file.flush()
                row_buffer = []
                
        if len(row_buffer) > 0:   
            logger.info(f'Row buffer has {len(row_buffer)} rows, writing to {output_fpath}.')                            
            pd.concat(row_buffer).to_csv(results_file, sep='\t',
                            index=False,
                            header=first_write)
            result_count += len(row_buffer)
        logger.info(f"Finished writing to {output_fpath} with {result_count} results")
            

def get_gene_list(args: argparse.Namespace) -> List[str]:
    """Get list of genes from either command line or file."""
    if args.genes:
        logger.info('Using genes from command line argument')
        return [gene.strip() for gene in args.genes.split(',')]
    elif args.gene_file:
        logger.info(f'Reading genes from file: {args.gene_file}')
        try:
            with open(args.gene_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f'Failed to read gene file: {e}')
            raise
    return []

def main():
    """Main entry point for the application"""
    args = setup_argparser()
    
    # Load config from file if specified, otherwise from args
    if args.config_file:
        config = MRConfig.from_file(args.config_file)
        config_from_file = True
    else:
        config = MRConfig.from_args(args)
        config_from_file = False

    # Create output directory
    config.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = config.output_prefix.joinpath('mr_miner.log')
    setup_logger(log_path, logging.DEBUG if config.verbose else logging.INFO)
    logger.info(f'MRMiner version {__version__} started')

    # Log config source
    if config_from_file:
        logger.info('Using configuration from file: %s', args.config_file)
    else:
        logger.info('Using configuration from command line arguments')

    logger.info('Configuration: %s', config)
    
    # Save config for reproducibility
    config_path = config.output_prefix.parent.joinpath('mr_miner.config')
    config.save(config_path)

    cache_path = Path(config.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Initialize ChromMapper
    chrom_mapper = ChromMapper(mapping_tsv_fpath=config.cache_path.joinpath('chromosome_mappings.tsv'))  # Add necessary initialization parameters
    # Initialize PlinkWrapper
    data_1kg_template = str(Path(config.thousand_genomes_basepath).joinpath("{pop}"))
    plink = PlinkWrapper(
        data_1kg_prefix_template=data_1kg_template,
        pop=config.pop,
        plink_fpath=config.plink_fpath
    )
    logger.info(f"Initialized PlinkWrapper with data_1kg_prefix_template: {data_1kg_template}")
    # Initialize SPDI translator based on configuration
    spdi_translator = initialize_spdi_translator(config, plink, chrom_mapper)
    # Initialize components
    variant_corrs = VariantCorrelations(
        spdi_rsid_translator=spdi_translator,
        plink_wrapper=plink,
        pop=config.pop
    )
    logger.info(f"Initialized VariantCorrelations with spdi_rsid_translator: {spdi_translator}")
    variant_proxies = VariantProxies(
        plink_wrapper=plink,
        cache_path=config.cache_path,
        r2_threshold=config.proxy_r2_threshold,
        distance_threshold_kb=config.proxy_distance_threshold_kb
    )
    logger.info(f"Initialized VariantProxies with cache file: {variant_proxies.cache_fpath}")
   
    all_exposure_data = data_wrappers.GtexEqtls(gtex_eqtl_path=config.eqtl_data_path, spdi_translator=spdi_translator)
    all_outcome_data = data_wrappers.OpentargetsResults.from_file(file=config.outcome_data_path,                                                                
                                                                  file_format="tsv",
                                                                   lead_variants_only=True)

    mr_data = data_wrappers.MRData(exposure_data=all_exposure_data,
                                  outcome_data=all_outcome_data,
                                  variant_corrs=variant_corrs,
                                  variant_proxies=variant_proxies,
                                  spdi_translator=spdi_translator)

    gene_list = get_gene_list(args)
    if not gene_list:
        gene_list = mr_data.exposure_data.all_genes
        logger.info(f'No gene list provided, using all genes in exposure data: {len(gene_list)} genes')
    else:
        logger.info(f'Using provided gene list: {len(gene_list)} genes')

    output_fpath = config.output_prefix.parent.joinpath('mr_miner_results.tsv.gz')
    run_mr_miner(mr_data, gene_list, output_fpath)


if __name__ == '__main__':
    main()
