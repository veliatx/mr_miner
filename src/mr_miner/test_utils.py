from pathlib import Path
import subprocess
import logging


logger = logging.getLogger(__name__)


def create_test_dbsnp_files(
    source_dir_path: Path | str,
    output_dir_path: Path | str,
    lines_per_file: int = 1000,
    num_files: int = 3
) -> None:
    """
    Create truncated copies of dbSNP JSON files for testing.
    
    Args:
        source_dir_path: Directory containing source dbSNP JSON.bz2 files
        output_dir_path: Directory to write test files
        lines_per_file: Number of JSON lines to keep per file
        num_files: Number of chromosome files to process
    """
    source_dir_path = Path(source_dir_path)
    output_dir_path = Path(output_dir_path)
    
    logger.info(f"Source directory: {source_dir_path}")
    logger.info(f"Output directory: {output_dir_path}")
    
    # Create output directory
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir_path}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise
        
    if not output_dir_path.exists():
        raise RuntimeError(f"Output directory does not exist after creation attempt: {output_dir_path}")
    
    # Find chromosome JSON files
    chr_files = sorted([
        f for f in source_dir_path.glob("refsnp-chr*.json.bz2")
        if not f.name.endswith(".processed.txt")
    ])[:num_files]
    
    if not chr_files:
        raise ValueError(f"No chromosome JSON files found in {source_dir_path}")
    
    logger.info(f"Found {len(chr_files)} source files")
    
    for src_file in chr_files:
        out_file = output_dir_path.joinpath(src_file.name)
        logger.info(f"Processing {src_file.name}")
        
        # Use bzcat and head to get first N lines, maintaining bzip2 compression
        cmd = f"bzcat {src_file} | head -n {lines_per_file} | bzip2 > {out_file}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Ran command: {cmd}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            continue
        
        # Verify file was created and contains data
        if not out_file.exists():
            logger.error(f"Output file was not created: {out_file}")
            continue
            
        if out_file.stat().st_size == 0:
            logger.error(f"Output file is empty: {out_file}")
            continue
            
        logger.info(f"Created {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")