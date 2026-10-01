"""Data quality engine: checks the raw tables, quarantines CRITICAL/HIGH records with lineage, and scores quality.

RAW - QUARANTINED = VALIDATED, exactly: nothing is silently edited or dropped.
Run: `python -m prime_cost.quality.validation`.
"""

from __future__ import annotations

import json

import pandas as pd

from prime_cost.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, load_settings

SEVERITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.3, "LOW": 0.1}
QUARANTINE_SEVERITIES = ("CRITICAL", "HIGH")


def _issues(dataset: str, ids, rule: str, severity: str, reason: str) -> pd.DataFrame:
    ids = pd.Series(ids)
    return pd.DataFrame({"dataset": dataset, "record_id": ids.astype(str).to_numpy(), "rule": rule, "severity": severity, "reason": reason})


def check_tickets(tickets: pd.DataFrame, lines: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    t = tickets.reset_index(names="_row")
    dup = t.duplicated("ticket_id", keep="first")
    out.append(_issues("tickets", t.loc[dup, "_row"], "duplicate_ticket", "HIGH", "same ticket_id appears more than once"))
    out.append(_issues("tickets", t.loc[t["channel"].isna(), "_row"], "missing_channel", "MEDIUM", "channel is empty"))
    hour_of_open = t["opened_at"].dt.hour
    ahead = (t["opened_at"].dt.normalize() != t["date"]) & (hour_of_open > 3)
    odd_hour = (hour_of_open < 12) & (hour_of_open > 3)
    out.append(_issues("tickets", t.loc[ahead | odd_hour, "_row"], "timestamp_outside_opening_hours", "HIGH", "ticket opened outside trading hours"))
    out.append(_issues("tickets", t.loc[t["covers"] <= 0, "_row"], "non_positive_covers", "CRITICAL", "covers must be at least 1"))
    orphan = ~t["ticket_id"].isin(lines["ticket_id"])
    out.append(_issues("tickets", t.loc[orphan, "_row"], "ticket_without_lines", "LOW", "ticket has no lines"))
    return out


def check_lines(lines: pd.DataFrame, tickets: pd.DataFrame, items: pd.DataFrame, price_history: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    ln = lines.reset_index(names="_row")
    out.append(_issues("ticket_lines", ln.loc[ln["qty"] <= 0, "line_id"], "non_positive_qty", "CRITICAL", "quantity must be positive"))
    out.append(_issues("ticket_lines", ln.loc[~ln["item_id"].isin(items["item_id"]), "line_id"], "unknown_item", "CRITICAL", "item_id not in the menu"))
    out.append(_issues("ticket_lines", ln.loc[~ln["ticket_id"].isin(tickets["ticket_id"]), "line_id"], "orphan_line", "CRITICAL", "ticket_id not in tickets"))
    ph = price_history.rename(columns={"price": "menu_price"})
    m = ln.merge(ph, on=["date", "item_id"], how="left")
    ratio = m["unit_price"] / m["menu_price"]
    bad = (ratio < 0.7) | (ratio > 1.4)
    out.append(_issues("ticket_lines", m.loc[bad, "line_id"], "price_mismatch", "HIGH", "unit price differs from the menu price that day by more than 30%"))
    out.append(_issues("ticket_lines", ln.loc[(ln["discount_pct"] < 0) | (ln["discount_pct"] > 0.5), "line_id"], "invalid_discount", "HIGH", "discount outside 0-50%"))
    return out


def check_purchases(p: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    p = p.reset_index(names="_row")
    out.append(_issues("purchases", p.loc[(p["qty_received"] <= 0) | (p["agreed_price"] <= 0), "order_id"], "non_positive_value", "CRITICAL", "quantity or price not positive"))
    over = p["invoiced_price"] > p["agreed_price"] * 1.03
    out.append(_issues("purchases", p.loc[over, "order_id"], "invoice_above_agreed_price", "MEDIUM", "invoiced price more than 3% above the agreed price"))
    short = p["qty_received"] < p["qty_ordered"] * 0.95
    out.append(_issues("purchases", p.loc[short, "order_id"], "short_delivery", "MEDIUM", "received less than 95% of what was ordered"))
    return out


def check_counts_shifts(counts: pd.DataFrame, shifts: pd.DataFrame, waste: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    c = counts.reset_index(names="_row")
    out.append(_issues("inventory_counts", c.loc[c["counted_qty"] < 0, "_row"], "negative_count", "CRITICAL", "counted quantity is negative"))
    s = shifts
    out.append(_issues("shifts", s.loc[(s["hours"] <= 0) | (s["hours"] > 14), "shift_id"], "invalid_hours", "CRITICAL", "shift length outside 0-14 hours"))
    w = waste.reset_index(names="_row")
    out.append(_issues("waste_log", w.loc[(w["qty"] <= 0) | (w["cost"] < 0), "_row"], "non_positive_waste", "CRITICAL", "waste quantity or cost invalid"))
    return out


def score(n_rows: dict, issues: pd.DataFrame) -> dict:
    """100 x (1 - weighted share of records with an issue), per dataset, then the simple mean."""
    per = {}
    for ds, n in n_rows.items():
        sub = issues[issues["dataset"] == ds]
        # a record counts once, with its worst severity
        worst = sub.assign(w=sub["severity"].map(SEVERITY_WEIGHT)).groupby("record_id")["w"].max()
        per[ds] = round(100.0 * (1.0 - worst.sum() / max(n, 1)), 2)
    return {"by_dataset": per, "overall": round(sum(per.values()) / len(per), 2)}


def run(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, output_dir=OUTPUT_DIR) -> dict:
    rd = lambda n: pd.read_parquet(raw_dir / f"{n}.parquet")  # noqa: E731
    tickets, lines, items = rd("tickets"), rd("ticket_lines"), rd("items")
    purchases, counts, shifts, waste = rd("purchases"), rd("inventory_counts"), rd("shifts"), rd("waste_log")
    price_history = rd("price_history")

    parts = (
        check_tickets(tickets, lines) + check_lines(lines, tickets, items, price_history)
        + check_purchases(purchases) + check_counts_shifts(counts, shifts, waste)
    )
    issues = pd.concat([p for p in parts if len(p)], ignore_index=True)

    # --- quarantine: CRITICAL/HIGH records only, each on its own. There is deliberately no cascade to the whole ticket:
    # that would remove ~3% of real sales and bias every later reconciliation of theoretical vs actual stock use.
    quar = issues[issues["severity"].isin(QUARANTINE_SEVERITIES)].copy()
    tickets_r = tickets.reset_index(names="_row")
    lines_r = lines.reset_index(names="_row")
    q_ticket_rows = set(quar.loc[quar["dataset"] == "tickets", "record_id"].astype(int))
    q_line_ids = set(quar.loc[quar["dataset"] == "ticket_lines", "record_id"])
    tickets_ok = tickets_r[~tickets_r["_row"].isin(q_ticket_rows)].drop(columns="_row")
    lines_ok = lines_r[~lines_r["line_id"].isin(q_line_ids)].drop(columns="_row")

    purchases_q = set(quar.loc[quar["dataset"] == "purchases", "record_id"])
    purchases_ok = purchases[~purchases["order_id"].isin(purchases_q)]
    counts_ok = counts[counts["counted_qty"] >= 0]
    waste_ok = waste[(waste["qty"] > 0) & (waste["cost"] >= 0)]
    shifts_ok = shifts[(shifts["hours"] > 0) & (shifts["hours"] <= 14)]

    n_rows = {"tickets": len(tickets), "ticket_lines": len(lines), "purchases": len(purchases), "inventory_counts": len(counts),
              "shifts": len(shifts), "waste_log": len(waste)}
    quality = score(n_rows, issues)
    by_rule = issues.groupby(["dataset", "rule", "severity"]).size().reset_index(name="records")
    report = {
        "score": quality, "rows": n_rows,
        "quarantined": {ds: int((quar["dataset"] == ds).sum()) for ds in n_rows},
        "validated": {"tickets": len(tickets_ok), "ticket_lines": len(lines_ok), "purchases": len(purchases_ok)},
        "issues_by_rule": by_rule.to_dict("records"),
        "issues_by_severity": issues.groupby("severity").size().to_dict(),
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {"tickets": tickets_ok, "ticket_lines": lines_ok, "purchases": purchases_ok, "inventory_counts": counts_ok,
                     "waste_log": waste_ok, "shifts": shifts_ok}.items():
        df.to_parquet(processed_dir / f"{name}_valid.parquet", index=False)
    quar.to_parquet(processed_dir / "quarantine.parquet", index=False)
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"data quality score {quality['overall']:.2f}/100; quarantined {len(quar)} records")
    return report


def main() -> None:
    load_settings()
    run()


if __name__ == "__main__":
    main()
