"""Tests for chromosome mapping."""
import pytest
from pathlib import Path
import logging

from mr_miner.chrom_mapper import ChromMapper


@pytest.fixture(scope="session")
def chrom_mapper_instance():
    """Fixture for ChromMapper instance."""
    return ChromMapper()


def test_chrom_mapper_initialization(chrom_mapper_instance):
    assert isinstance(chrom_mapper_instance, ChromMapper)


def test_get_chrom_length(chrom_mapper_instance):
    assert chrom_mapper_instance.get_length("chr1") == 248956422
    assert chrom_mapper_instance.get_length("chrUn") is None
    assert chrom_mapper_instance.get_length("invalid_chr") is None 