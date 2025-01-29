"""Tests for base classes."""
import pytest
from pathlib import Path
import logging

from mr_miner.base_classes import LoggedWrapper
from mr_miner.constants import DEFAULT_TRAIT_COLUMN


@pytest.fixture(scope="session")
def logged_wrapper():
    """Fixture for LoggedWrapper instance."""
    return LoggedWrapper(verbose=True)


def test_logged_wrapper_initialization(logged_wrapper):
    assert logged_wrapper.verbose is True
    assert isinstance(logged_wrapper.logger, logging.Logger)


def test_log_print(logged_wrapper, caplog):
    with caplog.at_level(logging.INFO):
        logged_wrapper.log_info("Test info message")
        assert "Test info message" in caplog.text


def test_default_trait_column():
    assert DEFAULT_TRAIT_COLUMN == "trait_reported" 