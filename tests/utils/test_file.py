# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import os

# Third Party Library
import pytest

# Private Library
from relay.utils.file import load_env_file

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _write_env(tmp_path, content: str):
    p = tmp_path / ".env"
    p.write_text(content)
    return p


class TestLoadEnvFile:
    def test_basic_key_value(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        p = _write_env(tmp_path, "MY_KEY=hello\n")
        load_env_file(p)
        assert os.environ["MY_KEY"] == "hello"

    def test_multiple_keys(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.delenv("KEY_B", raising=False)
        p = _write_env(tmp_path, "KEY_A=foo\nKEY_B=bar\n")
        load_env_file(p)
        assert os.environ["KEY_A"] == "foo"
        assert os.environ["KEY_B"] == "bar"

    def test_double_quoted_value_stripped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUOTED", raising=False)
        p = _write_env(tmp_path, 'QUOTED="my value"\n')
        load_env_file(p)
        assert os.environ["QUOTED"] == "my value"

    def test_single_quoted_value_stripped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SQUOTED", raising=False)
        p = _write_env(tmp_path, "SQUOTED='my value'\n")
        load_env_file(p)
        assert os.environ["SQUOTED"] == "my value"

    def test_comment_lines_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REAL_KEY", raising=False)
        p = _write_env(tmp_path, "# this is a comment\nREAL_KEY=yes\n")
        load_env_file(p)
        assert os.environ["REAL_KEY"] == "yes"

    def test_empty_lines_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPARSE_KEY", raising=False)
        p = _write_env(tmp_path, "\n\nSPARSE_KEY=found\n\n")
        load_env_file(p)
        assert os.environ["SPARSE_KEY"] == "found"

    def test_lines_without_equals_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VALID", raising=False)
        p = _write_env(tmp_path, "INVALID_LINE\nVALID=yes\n")
        load_env_file(p)
        assert os.environ.get("INVALID_LINE") is None
        assert os.environ["VALID"] == "yes"

    def test_setdefault_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXISTING", "original")
        p = _write_env(tmp_path, "EXISTING=new_value\n")
        load_env_file(p)
        assert os.environ["EXISTING"] == "original"

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.env"
        with pytest.raises(FileNotFoundError):
            load_env_file(missing)

    def test_accepts_str_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STR_PATH_KEY", raising=False)
        p = _write_env(tmp_path, "STR_PATH_KEY=works\n")
        load_env_file(str(p))
        assert os.environ["STR_PATH_KEY"] == "works"

    def test_empty_file_does_not_raise(self, tmp_path):
        p = _write_env(tmp_path, "")
        load_env_file(p)  # Should not raise

    def test_value_with_equals_sign_preserved(self, tmp_path, monkeypatch):
        monkeypatch.delenv("URL_KEY", raising=False)
        p = _write_env(tmp_path, "URL_KEY=http://example.com?a=1\n")
        load_env_file(p)
        assert os.environ["URL_KEY"] == "http://example.com?a=1"

    def test_whitespace_around_key_stripped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PADDED", raising=False)
        p = _write_env(tmp_path, "  PADDED  =value\n")
        load_env_file(p)
        assert os.environ["PADDED"] == "value"
