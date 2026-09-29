"""Offline release tests use a syntactic image identity for mocked Modal builders."""
import os

import pytest


@pytest.fixture(autouse=True)
def prepared_image_identity(monkeypatch):
    if 'FAYNT_BENCHMARK_IMAGE' not in os.environ:
        monkeypatch.setenv('FAYNT_BENCHMARK_IMAGE',
                          'registry.example/faynt-test@sha256:' + 'a' * 64)
