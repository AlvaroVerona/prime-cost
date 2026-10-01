"""Shared fixtures: one small synthetic bar (about 5 months) generated once per test session."""

from types import SimpleNamespace

import pytest

from prime_cost.analytics.facts import load_tables
from prime_cost.config import load_menu, small_settings
from prime_cost.data.generator import generate
from prime_cost.quality import validation


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("prime_cost")
    raw, proc, out = root / "raw", root / "processed", root / "outputs"
    settings = small_settings()
    gen = generate(settings, load_menu(), write=True, raw_dir=raw, processed_dir=proc)
    report = validation.run(raw, proc, out)
    tables = load_tables(raw, proc)
    return SimpleNamespace(settings=settings, gen=gen, report=report, tables=tables, raw=raw, proc=proc, out=out)
