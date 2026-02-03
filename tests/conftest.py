"""
Configuration for the unit tests.
"""

from pathlib import Path

import pytest


@pytest.fixture
def raw_csv_path() -> Path:
    return Path(__file__).parent / "fixtures" / "raw_data_mock.csv"
