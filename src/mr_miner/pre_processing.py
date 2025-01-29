import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Set
import pandas as pd
from tqdm import tqdm

from mr_miner.chrom_mapper import ChromMapper
from mr_miner.utilities import construct_vep_input_line, generate_spdi_str, iterate_cols, my_in1d, right_shift_variant_for_vep

logger = logging.getLogger(__name__)


def generate_opentargets_variants_file(opentargets_v2d_dir: Path, output_fpath: Path, chrom_mapper: ChromMapper):
    """
    Generate a file of variants from the opentargets V2D table.
    """
    
    seen_variants = set([])

    with gzip.open(output_fpath, 'wt') as output_file: 
        for chunk_counter, chunk_fpath in enumerate(tqdm(sorted(opentargets_v2d_dir.glob('*.parquet')))):
            print(f'Loading opentargets table chunk {chunk_counter} from {chunk_fpath} ...')
            
            this_chunk = pd.read_parquet(chunk_fpath)
            print(f'\tLoaded {this_chunk.shape[0]} rows and {this_chunk.shape[1]} columns.')
            
            this_chunk_variant_counter = 0
            
            for chrom, pos, ref, alt in iterate_cols(this_chunk, ('lead_chrom', 'lead_pos', 'lead_ref', 'lead_alt')):
                pos, ref, alt = right_shift_variant_for_vep(pos, ref, alt)
                # assert 3==4
                chrom = chrom_mapper.translate(chrom, 'plain', 'ucsc')
                
                if (chrom, pos, ref, alt) not in seen_variants:
                    seen_variants.add((chrom, pos, ref, alt))
                    chrom, start, end, refalt, strand = construct_vep_input_line(chrom, pos, ref, alt)
                    line = '\t'.join((chrom, str(start), str(end), refalt, strand)) 
                    output_file.write(line + '\n')
                    this_chunk_variant_counter += 1

            print(f'\tWrote {this_chunk_variant_counter} new variants to {output_fpath}. {len(seen_variants)} unique variants found so far.')

def merge_and_filter_opentargets_variants(opentargets_v2d_dir: Path, output_fpath: Path, chrom_mapper: ChromMapper):
    """
    Merge and filter the opentargets V2D table.
    """
    ot_variants  = set([])
    
    study_info = {}
    done_fpath = output_fpath.with_suffix('.done')
    log_fpath = output_fpath.with_suffix('.log')

    logger.info(f'Writing OpenTargets results to {output_fpath}.')
    
    if output_fpath.exists():
        output_fpath.unlink()

    if done_fpath.exists():
        done_fpath.unlink()

    if log_fpath.exists():
        log_fpath.unlink()    
    
    first_chunk = True
    
    for chunk_counter, chunk_fpath in enumerate(tqdm(list(opentargets_v2d_dir.glob('*.parquet')))):
        chunk_counter += 1
        logger.info(f'Loading opentargets table chunk {chunk_counter} from {chunk_fpath} ...')
        
        this_chunk = pd.read_parquet(chunk_fpath)
        logger.info(f'\tLoaded {this_chunk.shape[0]} rows and {this_chunk.shape[1]} columns.')
        # assess_cardinality(this_chunk, 'tag_pos', 'lead_pos')            

        study_info.update(extract_opentargets_study_info(this_chunk, display_tqdm=False))
       
        this_chunk = process_opentargets_chunk(this_chunk.copy(),
                                               variants_to_keep=None,
                                               chrom_source_dialect='plain',
                                               chrom_destination_dialect='ucsc')
        # assess_cardinality(this_chunk)
        num_variants_post_processing = len(set(this_chunk.lead_spdi))
        
        logger.info(f'\tAfter processing, chunk {chunk_counter} has {this_chunk.shape[0]} rows describing {num_variants_post_processing} unique variants with {this_chunk.shape[1]} columns.')
        
        ot_variants.update(this_chunk.lead_spdi)
        
        logger.info(f'\tFound {len(ot_variants)} unique post-filtered variants total so far...')
        with gzip.open(output_fpath, 'at') as output_file:
            this_chunk.to_csv(output_file, sep='\t', header=first_chunk, index=False)
            logger.info(f'\tWrote to {output_file.name}.')          
        first_chunk=False
        
    logger.info('All done.')
    done_fpath.touch()
    
    study_info = pd.DataFrame(study_info).T
    study_info.to_csv(output_fpath, sep='\t', index=True, compression='gzip')
    
    del(this_chunk)

FIELDS_TO_KEEP = ['lead_spdi', 'tag_spdi', 'chrom', 'pos', 'ref', 'alt', 'trait_reported', 'trait_efos', 'study_id', 'AFR_1000G_prop', 'AMR_1000G_prop', 'EAS_1000G_prop', 'EUR_1000G_prop', 
                  'SAS_1000G_prop', 'log10_ABF', 'trait_category', 'num_assoc_loci', 'posterior_prob', 'odds_ratio', 
                  'oddsr_ci_lower', 'oddsr_ci_upper', 'direction', 'beta', 'beta_ci_lower', 'beta_ci_upper', 'pval',
                  # 'lead_pos', 'tag_pos',
                  ]

def extract_opentargets_study_info(ot_df: pd.DataFrame,
                                  study_fields: List[str]=['pmid', 'pub_date', 'pub_journal', 'pub_title', 'pub_author', 'has_sumstats', 
                                                           'ancestry_initial', 'ancestry_replication', 'n_initial', 'n_replication', 'n_cases'],
                                  study_key_field: str='study_id',
                                  display_tqdm: bool=True
                                 ) -> Dict[str, Dict[str, Any]]:
    
    study_info = {}
    all_fields = [study_key_field] + study_fields

    if display_tqdm:
        iterator = tqdm(iterate_cols(ot_df, all_fields))
    else:
        iterator = iterate_cols(ot_df, all_fields)

    for values in iterator:
        this_key = values[0]
        if this_key not in study_info:
            study_info[this_key] = {k:v for k,v in zip(study_fields, values[1:])}
    return study_info


def process_opentargets_chunk(table_chunk: pd.DataFrame,
                              chrom_mapper: ChromMapper,
                              variants_to_keep=None, 
                              cols_to_keep=FIELDS_TO_KEEP,
                              remove_dups=False,
                              chrom_source_dialect='plain',
                              chrom_destination_dialect='ucsc',
                              seen_variants: Set[str]=set([])):
        
    table_chunk['tag_spdi'] =  [generate_spdi_str(chrom=chrom_mapper.translate_chrom_name(chrom, 
                                                                                 chrom_source_dialect, chrom_destination_dialect),
                                pos=pos-1, ref=ref, alt=alt) for chrom, 
                                                            pos, 
                                                            ref,
                                                            alt in iterate_cols(table_chunk, ('tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt'))]
    
    table_chunk['lead_spdi'] =  [generate_spdi_str(chrom=chrom_mapper.translate_chrom_name(chrom, 
                                                                                chrom_source_dialect, chrom_destination_dialect),
                                                    pos=pos-1, ref=ref, alt=alt) for chrom, 
                                                        pos, 
                                                        ref,
                                                        alt in iterate_cols(table_chunk, ('lead_chrom', 'lead_pos', 'lead_ref', 'lead_alt'))]
    
    print(f'\tBefore filtering, this chunk has information about {len(set(table_chunk.lead_spdi))} unique variants')
    seen_variants.update(table_chunk.lead_spdi)

    table_chunk['chrom'] = [chrom_mapper.translate_chrom_name(chrom, chrom_source_dialect, chrom_destination_dialect) for chrom in table_chunk['lead_chrom']]
    table_chunk['pos'] =  table_chunk.lead_pos
    table_chunk['ref'] = table_chunk.lead_ref
    table_chunk['alt'] = table_chunk.lead_alt        

    # Filter to only variants and columns of interest
    if cols_to_keep:
        table_chunk = table_chunk.loc[:, cols_to_keep]
    if variants_to_keep:
        table_chunk = table_chunk.loc[my_in1d(table_chunk['lead_spdi'].values, variants_to_keep)]
        
    # Convert EFO array to comma-separated string. May revert this in the future if we get more sophisticated in table relations
    trait_efos = []
    for entry in table_chunk.trait_efos:
        trait_efos.append(','.join(sorted(entry)))
    table_chunk['trait_efos'] = trait_efos
    
    # Remove duplicated rows
    if remove_dups:
        table_chunk = table_chunk.drop_duplicates(subset=['lead_spdi', 'trait_category',
                                                                  'trait_reported', 'trait_efos', 'study_id'],
                         keep='first')


    return table_chunk
