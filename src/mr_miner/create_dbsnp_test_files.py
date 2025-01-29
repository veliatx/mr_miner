#!/usr/bin/env python3
"""Create test dbSNP files by taking first N lines from source files."""
import argparse
import bz2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_test_files(source_path: Path, dest_path: Path, num_lines: int = 1000) -> None:
    """
    Create test files by copying first N lines from source files.
    
    Args:
        source_path: Path to directory containing source files
        dest_path: Path to directory where test files will be created
        num_lines: Number of lines to copy from each file
    """
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    source_files = list(source_path.glob('refsnp-chr*.json.bz2'))
    logger.info(f'Found {len(source_files)} source files')
    
    for source_file in source_files:
        dest_file = dest_path.joinpath(source_file.name)
        logger.info(f'Processing {source_file.name}...')
        
        with bz2.open(source_file, 'rt') as inf, bz2.open(dest_file, 'wt') as outf:
            for i, line in enumerate(inf):
                if i >= num_lines:
                    break
                outf.write(line)
                
        logger.info(f'Created {dest_file} with {min(num_lines, i+1)} lines')

def main():
    parser = argparse.ArgumentParser(description='Create test dbSNP files')
    parser.add_argument('source_path', type=Path, help='Path to source directory')
    parser.add_argument('dest_path', type=Path, help='Path to destination directory')
    parser.add_argument('-n', '--num_lines', type=int, default=1000,
                       help='Number of lines to copy from each file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    create_test_files(
        source_path=args.source_path,
        dest_path=args.dest_path,
        num_lines=args.num_lines
    )

if __name__ == '__main__':
    main()