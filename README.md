# mr_miner

Statistical genetics pipeline for performing Mendelian randomization efficiently at scale using public summary data. Currently contains classes to adapt OpenTargets GWAS data for outcomes and GTEX QTLs as exposures, but can be extended to other sources.

## Installation

```bash
pip install mr_miner
```

## Features

- Efficient variant ID translation between SPDI and rsID formats
- Tabix-indexed mapping files for fast lookups 
- Caching system for frequently accessed translations
- Multi-threaded processing of dbSNP JSON files
- Support for compressed input files (gzip, bzip2)


## Dependencies

- numpy
- pandas 
- scipy
- tqdm
- pyarrow
- statsmodels
- scikit-learn

## Development Setup

1. Clone the repository
2. Install development dependencies
3. Change to the source directory and run:
```bash
pip install .
```

## Testing

```bash
pytest
```

## Usage

```python
usage: mr_miner [-h] -o OUTPUT_PREFIX -t THOUSAND_GENOMES_BASEPATH -e ENSEMBL_GFF -c CACHE_PATH --eqtl_data_path EQTL_DATA_PATH
                [--outcome_data_path OUTCOME_DATA_PATH] (--dbsnp_vcf_fpath DBSNP_VCF_FPATH | --use_1kg_variants)
                [--genes GENES | --gene-file GENE_FILE] [-p POP] [-v] [-f CONFIG_FILE] [--plink_fpath PLINK_FPATH]
                [--proxy_r2_threshold PROXY_R2_THRESHOLD] [--proxy_distance_threshold_kb PROXY_DISTANCE_THRESHOLD_KB] [-@ THREADS]

Assess the causal relationship between a set of exposures and outcomes using public summary statistics.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        The prefix for the output files
  -t THOUSAND_GENOMES_BASEPATH, --thousand_genomes_basepath THOUSAND_GENOMES_BASEPATH
                        The basepath for the thousand genomes data
  -e ENSEMBL_GFF, --ensembl_gff ENSEMBL_GFF
                        Path to Ensembl GFF annotation file
  -c CACHE_PATH, --cache_path CACHE_PATH
                        Path to store cache files
  --eqtl_data_path EQTL_DATA_PATH
                        Path to directory containing GTEx eQTL data files
  --outcome_data_path OUTCOME_DATA_PATH
                        Path to directory containing OpenTargets variant info data
  --dbsnp_vcf_fpath DBSNP_VCF_FPATH
                        dbSNP VCF file
  --use_1kg_variants    Use 1000 Genomes variants for SPDI-rsID translation
  --genes GENES         Comma-separated list of gene names to analyze
  --gene-file GENE_FILE
                        Path to file containing gene names, one per line
  -p POP, --pop POP     Population code for LD calculations (default: EUR)
  -v, --verbose         Whether to run in verbose mode (show DEBUG messages)
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        Path to configuration file (overrides other arguments)
  --plink_fpath PLINK_FPATH
                        Path to plink executable
  --proxy_r2_threshold PROXY_R2_THRESHOLD
                        R² threshold for proxy variants (default: 0.9)
  --proxy_distance_threshold_kb PROXY_DISTANCE_THRESHOLD_KB
                        Distance threshold in kb for proxy variants (default: 250)
  -@ THREADS, --threads THREADS
                        Number of threads for multiprocessing (default: 64)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

Dylan Skola (dylan@veliatx.com)

For more details on implementation and usage, please refer to the documentation in the doc/ directory.
