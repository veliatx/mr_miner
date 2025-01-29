"""Tests for variant ID handling."""
import pytest
import gzip
from pathlib import Path
import logging
import subprocess
from unittest.mock import patch
import bz2
import json

from mr_miner.json_processor import process_json_file
from mr_miner.variant_ids import (
    SpdiRsidTranslatorDbSnp
)

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data paths
TEST_DATA_PATH = Path(__file__).parent.parent.joinpath("data").joinpath("dbsnp")

@pytest.fixture(scope="session")
def dbsnp_data_path():
    """Return path to test dbSNP data directory"""
    return TEST_DATA_PATH

@pytest.fixture(scope="session")
def cache_path(tmp_path_factory):
    """Create a temporary cache directory"""
    return tmp_path_factory.mktemp("cache")

def test_spdi_translator_initialization_and_caching(dbsnp_data_path, cache_path):
    """Test SpdiRsidTranslator initialization, map generation, and caching"""
    # Expected file paths based on cache directory
    spdi_map_fpath = cache_path.joinpath("spdi_to_rsid.txt.bgz")
    rsid_map_fpath = cache_path.joinpath("rsid_to_spdi.txt.bgz")
    spdi_map_cache_fpath = cache_path.joinpath("spdi_to_rsid_cache.txt.gz")
    rsid_map_cache_fpath = cache_path.joinpath("rsid_to_spdi_cache.txt.gz")

    
    # Initialize translator - should trigger map file generation
    translator = SpdiRsidTranslatorDbSnp(
        dbsnp_path=dbsnp_data_path,
        cache_path=cache_path,
        threads=1
    )
    
    # Verify both map files were created and indexed
    assert spdi_map_fpath.exists(), "SPDI map file not created"
    assert rsid_map_fpath.exists(), "rsID map file not created"
    assert Path(str(spdi_map_fpath) + '.tbi').exists(), "SPDI map file not indexed"
    assert Path(str(rsid_map_fpath) + '.tbi').exists(), "rsID map file not indexed"
    
    # Verify SPDI map file format using tabix
    tabix_output = subprocess.run(
        ["tabix", "-h", str(spdi_map_fpath), "NC_000001.11:1-1000000"],
        capture_output=True,
        text=True
    )
    assert tabix_output.returncode == 0, "SPDI map tabix query failed"
    
    for line in tabix_output.stdout.splitlines():
        fields = line.strip().split('\t')
        assert len(fields) == 4, f"Invalid field count in SPDI map: {line}"
        chrom, pos, spdi, rsid = fields
        assert chrom.startswith("NC_"), f"Invalid chromosome format: {chrom}"
        assert pos.isdigit(), f"Invalid position: {pos}"
        assert spdi.count(':') == 3, f"Invalid SPDI format: {spdi}"
        assert rsid.startswith("rs"), f"Invalid rsID format: {rsid}"
    
    # Get a real SPDI/rsID pair from the map file for testing
    with gzip.open(spdi_map_fpath, 'rt') as f:
        line = f.readline().strip()
        _, _, test_spdi, test_rsid = line.split('\t')
    
    logger.info(f"Testing with SPDI: {test_spdi} and rsID: {test_rsid}")
    
    # Test SPDI to rsID translation and caching
    rsids1 = translator.translate_spdis_to_rsids([test_spdi])
    
    assert spdi_map_cache_fpath.exists(), f"Spdi-to-rsid cache file not created after first query: {spdi_map_cache_fpath}"
    assert len(rsids1) == 1 and rsids1[0], "No translation found for test SPDI"
    
    # Test rsID to SPDI translation and caching
    spdis1 = translator.translate_rsids_to_spdis([test_rsid])
    assert rsid_map_cache_fpath.exists(), f"Rsid-to-spdi cache file not created after first query: {rsid_map_cache_fpath}"
    assert len(spdis1) == 1 and spdis1[0], "No translation found for test rsID"
    
    # Test cache hits
    rsids2 = translator.translate_spdis_to_rsids([test_spdi])
    spdis2 = translator.translate_rsids_to_spdis([test_rsid])
    assert rsids2 == rsids1, "Different result from cache (SPDI)"
    assert spdis2 == spdis1, "Different result from cache (rsID)"


def test_missing_translations(dbsnp_data_path, cache_path):
    """Test handling of missing translations."""
    translator = SpdiRsidTranslatorDbSnp(
        dbsnp_path=dbsnp_data_path,
        cache_path=cache_path,
        threads=1
    )
    
    # Test with invalid variants
    invalid_spdi = "NC_000001.11:999999999:X:Y"
    invalid_rsid = "rs999999999"
    
    # Test list mode (as_set=False)
    rsids = translator.translate_spdis_to_rsids([invalid_spdi])
    assert len(rsids) == 1, "Wrong result length for invalid SPDI"
    assert rsids[0] == "", "Invalid SPDI should translate to empty string"
    
    spdis = translator.translate_rsids_to_spdis([invalid_rsid])
    assert len(spdis) == 1, "Wrong result length for invalid rsID"
    assert spdis[0] == "", "Invalid rsID should translate to empty string"
    
    # Test set mode (as_set=True)
    rsid_set = translator.translate_spdis_to_rsids([invalid_spdi], as_set=True)
    assert len(rsid_set) == 0, "Invalid SPDI should give empty set"
    
    spdi_set = translator.translate_rsids_to_spdis([invalid_rsid], as_set=True)
    assert len(spdi_set) == 0, "Invalid rsID should give empty set"

def test_batch_translation(dbsnp_data_path, cache_path):
    """Test batch translation performance."""
    translator = SpdiRsidTranslatorDbSnp(
        dbsnp_path=dbsnp_data_path,
        cache_path=cache_path,
        threads=1
    )
    
    # Get test data
    test_pairs = []
    with gzip.open(cache_path.joinpath("spdi_to_rsid.txt.bgz"), 'rt') as f:
        for _ in range(10):  # Get 10 test pairs
            line = f.readline().strip()
            if not line:
                break
            _, _, spdi, rsid = line.split('\t')
            test_pairs.append((spdi, rsid))
    
    # Test batch translation
    spdis = [p[0] for p in test_pairs]
    rsids = [p[1] for p in test_pairs]
    
    # Should only make one tabix call each
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = ""  # Mock empty results since we're testing calls
        translator.translate_spdis_to_rsids(spdis)
        translator.translate_rsids_to_spdis(rsids)
        
        # Count tabix calls
        tabix_calls = sum(1 for call in mock_run.call_args_list 
                         if call.args[0][0] == 'tabix')
        assert tabix_calls == 2, "Expected exactly two tabix calls for batch translations"

def test_real_dbsnp_json_processing(dbsnp_data_path, tmp_path):
    """Test processing of real (truncated) dbSNP JSON files"""
    # Find all test JSON files
    json_files = list(dbsnp_data_path.glob("refsnp-chr*.json.bz2"))
    assert len(json_files) > 0, "No test JSON files found"
    
    # Process each file
    for json_file in json_files:
        logger.info(f"Processing {json_file}")
        spdi_to_rsid_fpath, rsid_to_spdi_fpath = process_json_file(
            json_fpath=json_file,
            output_path=tmp_path
        )
        
        # Verify spdi_to_rsid output format
        with open(spdi_to_rsid_fpath) as f:
            for line_num, line in enumerate(f, 1):
                fields = line.strip().split('\t')
                assert len(fields) == 4, f"Invalid field count in {json_file}, line {line_num}: {line}"
                chrom, pos, spdi, rsids = fields
                assert pos.isdigit(), f"Invalid position in {json_file}: {pos}"
                spdi_parts = spdi.split(':')
                assert len(spdi_parts) == 4, f"Invalid SPDI format in {json_file}: {spdi}"
                ref = spdi_parts[2]
                alt = spdi_parts[3]
                assert len(ref) > 0 or len(alt) > 0, f"Both alleles empty in {json_file}"
                assert all(r.startswith('rs') for r in rsids.split(',')), f"Invalid rsID format in {json_file}: {rsids}"
                
        # Verify rsid_to_spdi output format
        with open(rsid_to_spdi_fpath) as f:
            for line_num, line in enumerate(f, 1):
                fields = line.strip().split('\t')
                assert len(fields) == 3, f"Invalid field count in {json_file}, line {line_num}: {line}"
                rs, rsnum, spdis = fields
                assert rs == 'rs', f"Invalid rs prefix in {json_file}: {rs}"
                assert rsnum.isdigit(), f"Invalid rsID number in {json_file}: {rsnum}"
                for spdi in spdis.split(','):
                    parts = spdi.split(':')
                    assert len(parts) == 4, f"Invalid SPDI format in {json_file}: {spdi}"

def test_spdi_translator_with_real_dbsnp(dbsnp_data_path, cache_path):
    """Test full SpdiRsidTranslator workflow with real dbSNP data"""
    # Initialize translator with real data
    translator = SpdiRsidTranslatorDbSnp(
        dbsnp_path=dbsnp_data_path,
        cache_path=cache_path,
        threads=1
    )
    
    # Get some real SPDIs from the generated map file
    spdi_map_fpath = cache_path.joinpath("spdi_to_rsid.txt.bgz")
    test_spdis = set()
    with gzip.open(spdi_map_fpath, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            rsid = fields[3]  # rsID is fourth column
            if rsid.startswith('rs'):  # Only use entries with valid rsIDs
                test_spdis.add(fields[2])  # SPDI is third column
                if len(test_spdis) >= 5:  # Get 5 test SPDIs
                    break
    
    logger.info(f"Testing translation of SPDIs: {test_spdis}")
    
    # Test translation
    rsids = translator.translate_spdis_to_rsids(list(test_spdis))
    assert len(rsids) > 0, "No translations found"
    
    # Verify reverse translation
    for rsid in rsids:
        assert rsid.startswith("rs"), f"Invalid rsID format: {rsid}"
        spdis = translator.translate_rsids_to_spdis([rsid])
        assert len(set(spdis) & test_spdis) > 0, f"Reverse translation failed for {rsid}"

def test_dbsnp_json_content(dbsnp_data_path):
    """Test content and structure of test dbSNP JSON files"""
    for json_file in dbsnp_data_path.glob("refsnp-chr*.json.bz2"):
        logger.info(f"Checking {json_file}")
        with bz2.open(json_file, 'rt') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line)
                
                # Check required fields
                assert 'refsnp_id' in data, f"Missing refsnp_id in {json_file}, line {line_num}"
                assert 'primary_snapshot_data' in data, f"Missing primary_snapshot_data in {json_file}, line {line_num}"
                
                # Check placements
                placements = data['primary_snapshot_data']['placements_with_allele']
                assert len(placements) > 0, f"No placements in {json_file}, line {line_num}"
                
                # Check primary placement
                primary = next((p for p in placements if p.get('is_ptlp')), None)
                assert primary is not None, f"No primary placement in {json_file}, line {line_num}"
                
                # Check alleles
                assert len(primary['alleles']) > 0, f"No alleles in {json_file}, line {line_num}"
                for allele in primary['alleles']:
                    spdi = allele['allele']['spdi']
                    assert 'seq_id' in spdi, f"Missing seq_id in {json_file}, line {line_num}"
                    assert 'position' in spdi, f"Missing position in {json_file}, line {line_num}"
                    assert 'deleted_sequence' in spdi, f"Missing deleted_sequence in {json_file}, line {line_num}"
                    assert 'inserted_sequence' in spdi, f"Missing inserted_sequence in {json_file}, line {line_num}"

