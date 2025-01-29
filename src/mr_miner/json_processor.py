import collections
from pathlib import Path
import subprocess
import logging
from typing import Dict, Union, Tuple, List
import multiprocessing as mp
import gc

from mr_miner.constants import DBSNP_SPDI_POS_OFFSET


logger = logging.getLogger(__name__)


def process_json_file(json_fpath: Union[Path, str], output_path: Union[Path, str], 
                      chrom_translation_dict: Dict[str, str] | None = None, 
                      pos_offset: int=DBSNP_SPDI_POS_OFFSET) -> Tuple[Path, Path]:
    """Process a single dbSNP JSON file and extract SPDI mappings."""

    json_fpath = Path(json_fpath)
    output_path = Path(output_path)
    
    # Create temporary unsorted output files
    temp_spdi_to_rsid_fpath = output_path.joinpath(f"{json_fpath.stem}.spdi_to_rsid_raw.txt")
    temp_rsid_to_spdi_fpath = output_path.joinpath(f"{json_fpath.stem}.rsid_to_spdi_raw.txt")

    sorted_spdi_to_rsid_fpath = output_path.joinpath(f"{json_fpath.stem}.spdi_to_rsid.txt")
    sorted_rsid_to_spdi_fpath = output_path.joinpath(f"{json_fpath.stem}.rsid_to_spdi.txt")
    
    logger.info(f"Starting processing of {json_fpath}")

    # Get file size for logging
    file_size_mb = json_fpath.stat().st_size / (1024 * 1024)
    logger.info(f"Input file size: {file_size_mb:.2f} MB")

    # Determine compression type from filename
    suffix = json_fpath.suffix.lower()
    if suffix == '.gz':
        decompress_cmd = ['gzcat' if Path('/usr/bin/gzcat').exists() else 'zcat', str(json_fpath)]
    elif suffix == '.bz2':
        decompress_cmd = ['bzcat', str(json_fpath)]
    else:
        decompress_cmd = ['cat', str(json_fpath)]

    # Pipeline commands as lists to avoid shell injection
    jq_script = '''
        select(.refsnp_id != null) |
        .refsnp_id as $rs |
        .primary_snapshot_data.placements_with_allele[0].alleles[] |
        select(.allele.spdi != null) |
        .allele.spdi |
        select(.position != null and .deleted_sequence != null and .inserted_sequence != null) |
        select(.deleted_sequence != .inserted_sequence) |  # Skip reference alleles where del=ins
        [$rs, .seq_id, .position, .deleted_sequence, .inserted_sequence] |
        @tsv
    '''
    jq_cmd = ['jq', '-rc', jq_script]

    try:
        # Create pipeline
        p1 = subprocess.Popen(decompress_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(jq_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1.stdout.close()

        logger.info(f'Writing intermediate spdi-to-rsid map file {temp_spdi_to_rsid_fpath} ...')
        logger.info(f'Writing intermediate rsid-to-spdi map file {temp_rsid_to_spdi_fpath} ...')
        # Process output 
        line_count = 0
        with open(temp_spdi_to_rsid_fpath, 'wt') as spdi_to_rsid_raw_file, open(temp_rsid_to_spdi_fpath, 'wt') as rsid_to_spdi_raw_file:
            for line in p2.stdout:
                fields = line.decode().strip().split('\t')
                
                if len(fields) == 5:
                    rs_num, chrom, pos, ref, alt = fields
                    pos = int(pos) + pos_offset
                    rsid = 'rs' + rs_num
                    if chrom_translation_dict and chrom in chrom_translation_dict:
                        chrom = chrom_translation_dict[chrom]
                    
                    spdi = f'{chrom}:{pos}:{ref}:{alt}'
                    spdi_to_rsid_raw_file.write(f'{chrom}\t{pos}\t{spdi}\t{rsid}\n')
                    rsid_to_spdi_raw_file.write(f'rs\t{rs_num}\t{spdi}\n')
                    
                    line_count += 1
                    if line_count % 100000 == 0:
                        logger.info(f"Processed {line_count} lines of {json_fpath}")
                
        # Check for errors
        p2.stdout.close()

        for p, cmd in [(p1, decompress_cmd), (p2, jq_cmd)]:
            stderr = p.stderr.read().decode().strip()
            if stderr:
                logger.warning(f"Warning from {cmd[0]}: {stderr}")
            if p.wait() != 0:
                raise subprocess.CalledProcessError(p.returncode, cmd)
 
        logger.info(f"Successfully processed {json_fpath} ({line_count} lines)")
        
        # Sort the files
        logger.info(f"Sorting spdi_to_rsid file for {json_fpath}")
        sort_cmd = [
            "sort",
            "-k1,1",  # Sort by chromosome (seq_id)
            "-k2,2n",  # Sort numerically by position
            "-T", str(output_path),  # Use temp directory for sorting
            "-o", str(sorted_spdi_to_rsid_fpath),
            str(temp_spdi_to_rsid_fpath)
        ]
        subprocess.run(sort_cmd, check=True)
        
        logger.info(f"Sorting rsid_to_spdi file for {json_fpath}")
        sort_cmd = [
            "sort",
            "-k1,1",  # Sort by chromosome (seq_id)
            "-k2,2n",  # Sort numerically by position
            "-T", str(output_path),  # Use temp directory for sorting
            "-o", str(sorted_rsid_to_spdi_fpath),
            str(temp_rsid_to_spdi_fpath)
        ]
        subprocess.run(sort_cmd, check=True)
        
        # Clean up temporary files
        temp_spdi_to_rsid_fpath.unlink()
        temp_rsid_to_spdi_fpath.unlink()
        
        # After sorting, consolidate entries with same source ID
        logger.info(f"Consolidating spdi_to_rsid mappings for {json_fpath}")
        consolidated_spdi_to_rsid_fpath = output_path.joinpath(f"{json_fpath.stem}.spdi_to_rsid.consolidated.txt")
        with open(sorted_spdi_to_rsid_fpath, 'rt') as infile, open(consolidated_spdi_to_rsid_fpath, 'wt') as outfile:
            current_spdi = None
            current_chrom = None
            current_pos = None
            current_rsids = set()
            
            for line in infile:
                chrom, pos, spdi, rsid = line.strip().split('\t')
                
                if spdi != current_spdi:
                    # Write out previous group if it exists
                    if current_spdi is not None:
                        outfile.write(f'{current_chrom}\t{current_pos}\t{current_spdi}\t{",".join(sorted(current_rsids))}\n')
                    # Start new group
                    current_spdi = spdi
                    current_rsids = {rsid}
                    current_chrom = chrom
                    current_pos = pos
                else:
                    current_rsids.add(rsid)
            
            # Write last group
            if current_spdi is not None:
                outfile.write(f'{chrom}\t{pos}\t{current_spdi}\t{",".join(sorted(current_rsids))}\n')
        
        logger.info(f"Consolidating rsid_to_spdi mappings for {json_fpath}")
        consolidated_rsid_to_spdi_fpath = output_path.joinpath(f"{json_fpath.stem}.rsid_to_spdi.consolidated.txt")
        with open(sorted_rsid_to_spdi_fpath, 'rt') as infile, open(consolidated_rsid_to_spdi_fpath, 'wt') as outfile:
            current_rs_num = None
            current_spdis = set()
            
            for line in infile:
                _, rs_num, spdi = line.strip().split('\t')
                
                if rs_num != current_rs_num:
                    # Write out previous group if it exists
                    if current_rs_num is not None:
                        outfile.write(f'rs\t{current_rs_num}\t{",".join(sorted(current_spdis))}\n')
                    # Start new group
                    current_rs_num = rs_num
                    current_spdis = {spdi}
                else:
                    current_spdis.add(spdi)
            
            # Write last group
            if current_rs_num is not None:
                outfile.write(f'rs\t{current_rs_num}\t{",".join(sorted(current_spdis))}\n')
        
        # Clean up intermediate sorted files
        sorted_spdi_to_rsid_fpath.unlink()
        sorted_rsid_to_spdi_fpath.unlink()
        
        return consolidated_spdi_to_rsid_fpath, consolidated_rsid_to_spdi_fpath

    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {json_fpath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {json_fpath}: {e}")
        raise 

             