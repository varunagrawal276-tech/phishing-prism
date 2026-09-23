"""Tests for data loader utilities."""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import compute_md5, get_project_root, load_config


class TestProjectRoot:
    def test_finds_root(self):
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_root_has_src(self):
        root = get_project_root()
        assert (root / "src").is_dir()


class TestLoadConfig:
    def test_loads_data_config(self):
        config = load_config("data.yaml")
        assert "sources" in config
        assert "figshare" in config
        assert len(config["sources"]) == 7

    def test_sources_have_required_fields(self):
        config = load_config("data.yaml")
        for name, source in config["sources"].items():
            assert "file_id" in source, f"{name} missing file_id"
            assert "filename" in source, f"{name} missing filename"
            assert "md5" in source, f"{name} missing md5"

    def test_known_sources(self):
        config = load_config("data.yaml")
        expected = {"Ling", "Assassin", "Enron", "TREC-05", "TREC-06", "TREC-07", "CEAS-08"}
        assert set(config["sources"].keys()) == expected

    def test_invalid_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestComputeMD5:
    def test_md5_of_known_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        md5 = compute_md5(test_file)
        assert isinstance(md5, str)
        assert len(md5) == 32  # MD5 hex digest length
