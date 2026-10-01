import pandas as pd


def test_every_injected_issue_is_detected(pipeline):
    issues = pd.DataFrame(pipeline.report["issues_by_rule"]).groupby("rule")["records"].sum()
    inj = pipeline.gen["truth"]["injected_quality_issues"]
    assert issues["duplicate_ticket"] == inj["duplicate_ticket"]
    assert issues["missing_channel"] == inj["missing_channel"]
    assert issues["non_positive_qty"] == inj["negative_qty"]
    assert issues["price_mismatch"] >= inj["price_mismatch"] * 0.95  # a few overlap with negative quantities


def test_raw_minus_quarantined_equals_validated(pipeline):
    r = pipeline.report
    raw = pipeline.gen["tables"]
    assert r["rows"]["tickets"] - r["quarantined"]["tickets"] == r["validated"]["tickets"] == len(pipeline.tables["tickets"])
    assert r["rows"]["ticket_lines"] - r["quarantined"]["ticket_lines"] == r["validated"]["ticket_lines"]
    assert len(raw["ticket_lines"]) == r["rows"]["ticket_lines"]


def test_validated_tables_have_no_known_defects(pipeline):
    t = pipeline.tables
    assert not t["tickets"].duplicated("ticket_id").any()
    assert (t["ticket_lines"]["qty"] > 0).all()


def test_quarantine_has_lineage_columns(pipeline):
    q = pd.read_parquet(pipeline.proc / "quarantine.parquet")
    assert {"dataset", "record_id", "rule", "severity", "reason"} <= set(q.columns)
    assert q["severity"].isin(["CRITICAL", "HIGH"]).all()


def test_quality_score_is_between_0_and_100(pipeline):
    s = pipeline.report["score"]
    assert 90 <= s["overall"] <= 100
    assert all(0 <= v <= 100 for v in s["by_dataset"].values())
