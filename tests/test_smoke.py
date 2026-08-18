"""Smoke tests: package imports cleanly and core helpers behave.

These run without the OCR/GGUF server — they only verify that the modules
import and that pure helper functions work. Real OCR requires a running
llama.cpp server and is exercised manually (see README "Testing").
"""
import importlib

import pytest


@pytest.mark.parametrize("module", ["core.config", "core.pdf", "core.ocr"])
def test_module_imports(module):
    """Each core module should import without side effects (no server needed)."""
    assert importlib.import_module(module) is not None


def test_config_has_server_url_default():
    import core.config as cfg

    assert isinstance(cfg.OCR_SERVER_URL, str)
    assert cfg.OCR_SERVER_URL  # non-empty default
