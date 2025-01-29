from dataclasses import dataclass, asdict
from pathlib import Path
import configparser
from typing import Optional
import argparse

from mr_miner.constants import (
    DEFAULT_POP,
    DEFAULT_THREADS,
    DEFAULT_PROXY_R2_THRESHOLD,
    DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
)


@dataclass
class MRConfig:
    """Configuration for MR analysis pipeline"""

    # Required paths
    output_prefix: Path
    thousand_genomes_basepath: Path
    ensembl_gff: Path
    cache_path: Path
    eqtl_data_path: Path
    outcome_data_path: Path
    # Variant ID translation source (one must be provided)
    dbsnp_vcf_fpath: Optional[Path] = None
    use_1kg_variants: bool = False

    # Optional settings
    plink_fpath: Path = Path('plink')
    pop: str = DEFAULT_POP
    verbose: bool = False
    proxy_r2_threshold: float = DEFAULT_PROXY_R2_THRESHOLD
    proxy_distance_threshold_kb: float = DEFAULT_PROXY_DISTANCE_THRESHOLD_KB
    threads: int = DEFAULT_THREADS

    def __init__(self, 
                 output_prefix: Path,
                 thousand_genomes_basepath: Path,
                 ensembl_gff: Path,
                 cache_path: Path,
                 eqtl_data_path: Path,
                 outcome_data_path: Path,
                 dbsnp_vcf_fpath: Optional[Path] = None,
                 pop: str = DEFAULT_POP,
                 verbose: bool = False,
                 plink_fpath: Path = Path('plink'),
                 proxy_r2_threshold: float = DEFAULT_PROXY_R2_THRESHOLD,
                 proxy_distance_threshold_kb: float = DEFAULT_PROXY_DISTANCE_THRESHOLD_KB,
                 threads: int = DEFAULT_THREADS):
        self.output_prefix = Path(output_prefix)
        self.thousand_genomes_basepath = Path(thousand_genomes_basepath)
        self.ensembl_gff = Path(ensembl_gff)
        self.cache_path = Path(cache_path)
        self.dbsnp_vcf_fpath = Path(dbsnp_vcf_fpath) if dbsnp_vcf_fpath else None
        self.eqtl_data_path = Path(eqtl_data_path)
        self.outcome_data_path = Path(outcome_data_path)
        self.pop = pop
        self.verbose = verbose
        self.plink_fpath = Path(plink_fpath)
        self.proxy_r2_threshold = proxy_r2_threshold
        self.proxy_distance_threshold_kb = proxy_distance_threshold_kb
        self.threads = threads

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'MRConfig':
        """Create config from command line arguments."""
        return cls(
            output_prefix=args.output_prefix,
            dbsnp_vcf_fpath=args.dbsnp_vcf_fpath,
            thousand_genomes_basepath=args.thousand_genomes_basepath,
            ensembl_gff=args.ensembl_gff,
            cache_path=args.cache_path,
            eqtl_data_path=args.eqtl_data_path,
            outcome_data_path=args.outcome_data_path,
            pop=args.pop,
            verbose=args.verbose,
            plink_fpath=args.plink_fpath,
            proxy_r2_threshold=args.proxy_r2_threshold,
            proxy_distance_threshold_kb=args.proxy_distance_threshold_kb,
            threads=args.threads
        )

    @classmethod
    def from_file(cls, config_path: Path | str) -> 'MRConfig':
        """Create config from configuration file"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        return cls(
            output_prefix=Path(config['Paths']['output_prefix']),
            thousand_genomes_basepath=Path(config['Paths']['thousand_genomes_basepath']),
            ensembl_gff=Path(config['Paths']['ensembl_gff']),
            cache_path=Path(config['Paths']['cache_path']),
            eqtl_data_path=Path(config['Paths']['eqtl_data_path']),
            outcome_data_path=Path(config['Paths']['outcome_data_path']),
            dbsnp_vcf_fpath=Path(config['Paths'].get('dbsnp_vcf_fpath')) if 'dbsnp_vcf_fpath' in config['Paths'] else None,
            use_1kg_variants=config['Parameters'].getboolean('use_1kg_variants', False),
            plink_fpath=Path(config['Paths'].get('plink_fpath', 'plink')),
            pop=config['Parameters'].get('pop', DEFAULT_POP),
            verbose=config['Parameters'].getboolean('verbose', False),
            proxy_r2_threshold=config['Parameters'].getfloat('proxy_r2_threshold', DEFAULT_PROXY_R2_THRESHOLD),
            proxy_distance_threshold_kb=config['Parameters'].getfloat('proxy_distance_threshold_kb', DEFAULT_PROXY_DISTANCE_THRESHOLD_KB),
            threads=config['Parameters'].getint('threads', DEFAULT_THREADS)
        )

    def save(self, output_path: Path | str) -> None:
        """Save configuration to file"""
        config = configparser.ConfigParser()
        
        # Convert dataclass to dict and split into sections
        config_dict = asdict(self)
        config['Paths'] = {
            k: str(v) if v is not None else '' 
            for k, v in config_dict.items() 
            if k in [
                'output_prefix', 'thousand_genomes_basepath', 'ensembl_gff',
                'cache_path', 'eqtl_data_path', 'dbsnp_vcf_fpath', 'plink_fpath'
            ]
        }
        config['Parameters'] = {
            k: str(v) 
            for k, v in config_dict.items()
            if k in [
                'use_1kg_variants', 'pop', 'verbose', 'proxy_r2_threshold', 
                'proxy_distance_threshold_kb', 'threads'
            ]
        }
        
        with open(output_path, 'w') as f:
            config.write(f)
