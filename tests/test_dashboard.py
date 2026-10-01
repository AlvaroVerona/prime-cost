"""Every dashboard page renders from the committed pipeline outputs."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "app.py", *sorted((APP / "views").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def test_overview_shows_the_headline_numbers():
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=120)
    at.run()
    labels = [m.label for m in at.metric]
    assert "EBITDA margin" in labels and any("Data quality" in x for x in labels)
