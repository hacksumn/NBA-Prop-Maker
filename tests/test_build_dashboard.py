"""Tests for scripts/build_dashboard.py (read-only Command Center dashboard).

The dashboard reports on real-money results, so these tests pin the numbers (record, units,
net dollars), the integrity flags, and the read-only guarantee against a small synthetic
project tree.
"""
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

import scripts.build_dashboard as bd

NOW = datetime(2026, 11, 15, 9, 0, 0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _pick(date, player, prop, result, price=None, dec=None, line=4.5, **kw):
    row = {"game_date": date, "player": player, "team": "CLE", "opponent": "DAL", "prop": prop,
           "direction": "UNDER", "line": line, "result": result, "actual": None,
           "side_price": price, "side_decimal_odds": dec, "ev_per_unit": 0.05,
           "model_side_prob": 0.6, "no_vig_side_prob": 0.52, "confidence": 66.0, "signal_grade": "C",
           "publish_tier": "core", "published_at": f"{date} 08:00:00", "line_source": "fanduel",
           "prediction": 3.9, "l10_avg": 4.1}
    row.update(kw)
    return row


def make_project(root: Path) -> None:
    """Build a tiny project: 2026-27 season picks, a backfill pick, an all-DNP date, slips, logs."""
    out = root / "output"
    picks = [
        _pick("2026-11-01", "Alpha One", "AST", "WIN", price=-110),               # priced via American odds
        _pick("2026-11-01", "Bravo Two", "STL", "LOSS", dec=1.5, line=1.5, team="DEN", opponent="NOP"),  # priced via decimal odds
        _pick("2026-11-02", "Alpha One", "AST", "WIN", price=100, line=3.5),
        _pick("2026-11-02", "Charlie Three", "BLK", "WIN", line=1.5, team="BKN", opponent="MEM"),  # unpriced: no unit P&L
        _pick("2026-11-03", "Delta Four", "AST", "DNP", price=-120),
        _pick("2026-11-04", "Echo Five", "AST", None, price=-115, line=6.5),       # pending
        _pick("2026-11-04", "Evil </script><b>x", "AST", None, price=-105, line=2.5),
        _pick("2026-03-20", "Old Timer", "PTS", "WIN", line=20.5, line_source=None),  # backfill era
    ]
    picks += [_pick("2026-04-11", f"Dnp Guy {i}", "AST", "DNP") for i in range(3)]  # all-DNP date
    _write_csv(out / "picks_history.csv", picks)
    _write_csv(out / "picks_latest.csv", [_pick("2026-11-04", "Echo Five", "AST", None, price=-118, line=6.5)
                                          | {"commence_time": "2026-11-05T00:10:00Z", "sigma": 2.5, "meta_prob": 55.0}])

    def slip(date, typ, result, payout, legs):
        r = {"game_date": date, "slip_type": typ, "wager": 20.0, "potential_payout": 60.0,
             "actual_payout": payout, "result": result, "joint_prob_adj": 0.4, "ev_per_unit": 0.1}
        for i, (pl, pr) in enumerate(legs, 1):
            r.update({f"Pick_{i}_Player": pl, f"Pick_{i}_Prop": pr, f"Pick_{i}_Dir": "UNDER", f"Pick_{i}_Line": 4.5})
        return r
    legs_a = [("Alpha One", "AST"), ("Bravo Two", "STL")]
    legs_b = [("Alpha One", "AST"), ("Delta Four", "AST")]
    slips = [
        slip("2026-11-01", "2-pick Power Play", "WIN", 60.0, legs_a),
        slip("2026-11-01", "2-pick UNDER-Only", "WIN", 60.0, legs_a),   # same legs, second label
        slip("2026-11-02", "2-pick Power Play", "LOSS", 0.0, legs_b),
        slip("2026-11-03", "2-pick Power Play", "PUSH", 20.0, legs_b),
        {"result": "LOSS", "actual_payout": 0.0},                         # orphan: no date, no legs
    ]
    _write_csv(out / "betslips_history.csv", slips)

    cands = []
    for bucket, fail, res, prof in [("published", "tier selection/diversity", "WIN", 0.9),
                                    ("published", "tier selection/diversity", "LOSS", -1.0),
                                    ("blocked_price", "price: price -270 beyond -200 cap", "WIN", 0.4),
                                    ("blocked_other", "meta 38.8<42.0", "LOSS", -1.0)]:
        cands.append({"game_date": "2026-11-02", "prop": "AST", "player": "P", "result": res, "side_decimal_odds": 1.9,
                      "profit_units": prof, "model_side_prob": 0.62, "no_vig_side_prob": 0.5, "ev_per_unit": 0.1,
                      "gate_bucket": bucket, "gate_failures": fail})
    _write_csv(out / "candidates_history.csv", cands)

    (out / "DECISIONS.md").write_text(
        "# DECISIONS\n\n## [DEC-001] Use chronological splits\n- Date: (pre-April 2026)\n- Status: Accepted\n\n"
        "## DEC-047 — Spend the reserve\n\n## 2026-11-13 — DEC-077 — Carry season_games_played\n\n"
        "## DEC-075 ? Repair timestamp integrity\n\n## 2026-11-10 ? Isolated-refresh path correction (DEC-076)\n",
        encoding="utf-8")
    (out / "TASKS.md").write_text(
        "# TASKS\n\n## NOW\n\n- [ ] **[NEEDS JAKE — DEC-082] Backfill the box scores.** Then `decide`.\n"
        "- [x] finished thing\n- [ ] plain open item\n\n## NEXT\n\n- [ ] not in NOW\n", encoding="utf-8")

    data = root / "data"
    _write_csv(data / "nba_data.csv", [{"game_date": "2026-11-14", "player": "Alpha One", "season": "2026-27"}])
    _write_csv(data / "historical_lines.csv", [{"game_date": "2026-11-14", "source": "fanduel"}])

    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "run_20261114.log").write_text(
        "2026-11-14 05:37:23,839 [INFO]   STEP 1 — NBA Game Logs  (nba_api → data/nba_data.csv)\n"
        "2026-11-14 05:38:05,000 [INFO]   STEP 2 — Fanduel Snapshot  (authoritative)\n"
        "2026-11-14 05:38:06,000 [INFO]     Fetched 307 validated player props (fanduel, 5/5 events, 19001 credits before prop requests)\n"
        "2026-11-14 05:38:07,000 [WARNING]   something odd\n"
        "2026-11-14 05:45:52,000 [INFO]   Log saved to: x\n"
        # >20 minute gap, then the next run's banner lines BEFORE its STEP 1 line
        "2026-11-14 20:53:54,000 [INFO]   ====\n"
        "2026-11-14 20:53:54,100 [INFO]   banner\n"
        "2026-11-14 20:53:54,329 [INFO]   STEP 1 — NBA Game Logs  (x)\n"
        "2026-11-14 20:54:34,000 [INFO]   STEP 2 — Fanduel Snapshot  (x)\n"
        "2026-11-14 20:54:35,000 [INFO]     All 4 event(s) on 2026-11-14 have already started; 18858 provider credits remaining\n"
        "2026-11-14 20:54:35,500 [ERROR]   [FAIL] Fanduel fetch failed: no current fanduel player-prop lines returned\n"
        "2026-11-14 20:54:36,000 [INFO]   STEP 3 — Layer 5\n", encoding="utf-8")


def _hash_tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class HelperTests(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = bd.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(bd.wilson(0, 0), (None, None))
        lo, hi = bd.wilson(10, 10)
        self.assertLess(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_american_to_decimal(self):
        self.assertAlmostEqual(bd.american_to_decimal(-110), 1.909090909, places=6)
        self.assertAlmostEqual(bd.american_to_decimal(120), 2.2)
        self.assertAlmostEqual(bd.american_to_decimal(-200), 1.5)
        for bad in (None, 0, float("nan"), "abc"):
            self.assertIsNone(bd.american_to_decimal(bad))

    def test_era_boundaries(self):
        d = pd.Series(["2026-03-17", "2026-03-29", "2026-03-30", "2026-05-10", "2026-05-11", "2026-09-30", "2026-10-26"])
        self.assertEqual(list(bd.assign_era(d)), ["backfill", "backfill", "live_pp", "live_pp", "live_fd", "live_fd", "season"])

    def test_gate_category(self):
        self.assertEqual(bd.gate_category("published", "tier selection/diversity"), "Published")
        self.assertEqual(bd.gate_category("blocked_price", "price: price -270 beyond -200 cap"), "Favorite cap")
        self.assertEqual(bd.gate_category("blocked_price", "price: EV -0.054<+0.010 at -140"), "Min EV")
        self.assertEqual(bd.gate_category("blocked_other", "meta 38.8<42.0"), "Meta floor")
        self.assertEqual(bd.gate_category("blocked_other", "vol 3.30>2.92"), "Volatility")
        self.assertEqual(bd.gate_category("blocked_other", "tier selection/diversity"), "Tier / diversity")
        self.assertEqual(bd.gate_category("blocked_other", None), "Other")

    def test_clean_removes_nan_and_numpy(self):
        import numpy as np
        out = bd.clean({"a": float("nan"), "b": np.int64(3), "c": [np.float64(1.5), float("inf")], "d": np.bool_(True)})
        self.assertEqual(out, {"a": None, "b": 3, "c": [1.5, None], "d": True})
        json.dumps(out, allow_nan=False)


class ParserTests(unittest.TestCase):
    def test_decision_heading_styles(self):
        with tempfile.TemporaryDirectory() as td:
            make_project(Path(td))
            ds = {d["id"]: d for d in bd.parse_decisions(Path(td) / "output" / "DECISIONS.md")}
        self.assertEqual(set(ds), {"DEC-001", "DEC-047", "DEC-075", "DEC-076", "DEC-077"})
        self.assertEqual(ds["DEC-077"]["date"], "2026-11-13")
        self.assertEqual(ds["DEC-076"]["title"], "Isolated-refresh path correction")  # no dangling "()"
        self.assertEqual(ds["DEC-001"]["status"], "Accepted")
        self.assertEqual(list(ds)[0], "DEC-077")  # newest id first

    def test_tasks_only_from_now_section(self):
        with tempfile.TemporaryDirectory() as td:
            make_project(Path(td))
            tasks = bd.parse_tasks(Path(td) / "output" / "TASKS.md")
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["tag"], "NEEDS JAKE — DEC-082")
        self.assertNotIn("`", tasks[0]["text"])
        self.assertEqual([t["done"] for t in tasks], [False, True, False])

    def test_run_duration_stops_at_gap_and_ignores_next_banner(self):
        with tempfile.TemporaryDirectory() as td:
            make_project(Path(td))
            runs = bd._parse_log(Path(td) / "logs" / "run_20261114.log")
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["end"], "2026-11-14 05:45:52")  # not 20:53 (next run's banner)
        self.assertEqual(len(runs[0]["steps"]), 2)
        self.assertEqual(runs[1]["steps"][1]["err"], 1)
        self.assertTrue(runs[1]["steps"][1].get("tipped"))


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_project(cls.root)
        cls.before = _hash_tree(cls.root)
        cls.payload = bd.build_payload(cls.root, now=NOW)
        cls.after = _hash_tree(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_is_read_only(self):
        self.assertEqual(self.before, self.after)

    def test_season_record_and_units(self):
        s = self.payload["scopes"]["season"]["summary"]
        self.assertEqual((s["w"], s["l"], s["dnp"], s["pending"]), (3, 1, 1, 2))
        self.assertEqual(s["priced_n"], 3)  # the unpriced BLK win never counts toward units
        self.assertAlmostEqual(s["units"], (100 / 110) + (-1.0) + 1.0, places=5)
        self.assertAlmostEqual(s["hit"], 0.75)
        self.assertAlmostEqual(s["priced_hit"], 2 / 3, places=5)

    def test_unpriced_pick_never_gets_guessed_units(self):
        rows = {r["pl"]: r for r in self.payload["picks"]}
        charlie = rows["Charlie Three"]                      # a WIN with no logged price
        self.assertEqual(charlie["rs"], "WIN")
        self.assertIsNone(charlie["dc"])
        self.assertIsNone(charlie["u"])
        alpha = [r for r in self.payload["picks"] if r["pl"] == "Alpha One" and r["d"] == "2026-11-01"][0]
        self.assertAlmostEqual(alpha["u"], 100 / 110, places=5)  # priced from the American odds alone
        self.assertEqual(rows["Delta Four"]["rs"], "DNP")
        self.assertIsNone(rows["Delta Four"]["u"])             # DNP is void, never a loss

    def test_scopes_and_eras(self):
        sc = self.payload["scopes"]
        self.assertEqual(sc["live"]["summary"]["w"], 3)          # live excludes the backfill win
        self.assertEqual(sc["all"]["summary"]["w"], 4)           # all includes it
        eras = {e["key"]: e for e in self.payload["eras"]}
        self.assertEqual(eras["backfill"]["w"], 1)
        self.assertEqual(eras["season"]["l"], 1)

    def test_equity_and_streak(self):
        days = self.payload["scopes"]["season"]["series"]["days"]
        self.assertEqual([d["date"] for d in days], ["2026-11-01", "2026-11-02"])
        self.assertAlmostEqual(days[0]["units"], (100 / 110) - 1.0, places=5)
        self.assertAlmostEqual(days[-1]["cum_units"], self.payload["scopes"]["season"]["summary"]["units"], places=5)
        self.assertEqual(self.payload["scopes"]["season"]["series"]["streak"], {"type": "W", "len": 1})
        self.assertGreater(self.payload["scopes"]["season"]["series"]["max_drawdown"], 0)

    def test_slips_orphan_and_duplicates(self):
        sl = self.payload["slips"]
        self.assertEqual(sl["orphans"]["n"], 1)
        self.assertEqual(sl["orphans"]["csv_lines"], [6])        # 5 data rows + header; orphan is the 5th data row
        self.assertEqual(sl["raw"]["n"], 4)
        self.assertEqual(sl["unique"]["n"], 3)
        self.assertEqual(sl["duplicates"], 1)
        # unique: WIN +40, LOSS -20, PUSH 0
        self.assertAlmostEqual(sl["unique"]["net"], 20.0)
        self.assertEqual((sl["unique"]["w"], sl["unique"]["l"], sl["unique"]["p"]), (1, 1, 1))
        self.assertAlmostEqual(sl["raw"]["net"], 60.0)           # the duplicate WIN counted twice in the raw view
        kinds = [f["kind"] for f in self.payload["health"]["flags"]]
        self.assertIn("orphan_slip", kinds)

    def test_all_dnp_date_flagged(self):
        flags = [f for f in self.payload["health"]["flags"] if f["kind"] == "all_dnp"]
        self.assertEqual([f["date"] for f in flags], ["2026-04-11"])

    def test_today_card_and_dropped_pick(self):
        t = self.payload["today"]
        self.assertEqual(t["card_date"], "2026-11-04")
        self.assertEqual((t["n"], t["pending"]), (2, 2))
        self.assertEqual(t["dropped_from_latest"], ["Evil </script><b>x"])
        echo = next(p for p in t["picks"] if p["player"] == "Echo Five")
        self.assertTrue(echo["in_latest"])
        self.assertEqual(echo["latest_price"], -118.0)
        self.assertEqual(echo["price"], -115.0)

    def test_gates_buckets(self):
        b = {x["bucket"]: x for x in self.payload["gates"]["buckets"]}
        self.assertEqual((b["published"]["w"], b["published"]["l"]), (1, 1))
        self.assertAlmostEqual(b["published"]["units"], -0.1, places=5)
        cats = {c["category"] for c in self.payload["gates"]["categories"]}
        self.assertTrue({"Published", "Favorite cap", "Meta floor"} <= cats)

    def test_pipeline_runs_and_late_status(self):
        p = self.payload["health"]["pipeline"]
        self.assertEqual(p["n_runs"], 2)
        first, last = p["recent_runs"][0], p["recent_runs"][1]
        self.assertEqual(first["status"], "warn")
        self.assertEqual(last["status"], "late")
        self.assertIn("DEC-068", last["note"])
        self.assertLess(first["secs"], 600)  # 05:37 -> 05:45, not hours
        self.assertEqual([c["credits"] for c in p["credits"]], [19001, 18858])

    def test_missing_sections_degrade_loudly_not_silently(self):
        warnings = self.payload["meta"]["warnings"]
        self.assertTrue(any("clv" in w.lower() for w in warnings))
        self.assertTrue(any("training_results" in w for w in warnings))
        self.assertEqual(self.payload["clv"], {})

    def test_decisions_and_tasks_present(self):
        self.assertEqual(len(self.payload["decisions"]), 5)
        self.assertEqual(len(self.payload["tasks"]), 3)

    def test_payload_is_strict_json(self):
        json.dumps(self.payload, allow_nan=False)

    def test_insights_cover_season_and_flags(self):
        titles = [i["title"] for i in self.payload["insights"]]
        self.assertIn("Season scoreboard", titles)
        self.assertIn("Data integrity flag", titles)
        self.assertIn("Latest pipeline run was a late rerun", titles)


class ProvisionalTests(unittest.TestCase):
    def test_provisional_results_never_enter_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            _write_csv(root / "output" / "clv_picks.csv", [
                {"game_date": "2026-11-04", "player": "Echo Five", "stat": "ast", "line": 6.5, "direction": "UNDER",
                 "result": "WIN", "actual": 4.0},
                {"game_date": "2026-11-04", "player": "Evil </script><b>x", "stat": "ast", "line": 2.5,
                 "direction": "UNDER", "result": "LOSS", "actual": 3.0},
                # an already-official pick must keep its official result even if the CLV file disagrees
                {"game_date": "2026-11-01", "player": "Alpha One", "stat": "ast", "line": 4.5, "direction": "UNDER",
                 "result": "LOSS", "actual": 9.0},
            ])
            payload = bd.build_payload(root, now=NOW)
        t = payload["today"]
        self.assertEqual(t["provisional"], {"n": 2, "w": 1, "l": 1, "p": 0})
        by = {p["player"]: p for p in t["picks"]}
        self.assertEqual(by["Echo Five"]["prov"], {"result": "WIN", "actual": 4.0})
        self.assertIsNone(by["Echo Five"]["result"])             # official result still pending
        # official season record is untouched by the provisional rows
        s = payload["scopes"]["season"]["summary"]
        self.assertEqual((s["w"], s["l"], s["pending"]), (3, 1, 2))

    def test_no_clv_file_means_no_provisional(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            t = bd.build_payload(root, now=NOW)["today"]
        self.assertEqual(t["provisional"]["n"], 0)
        self.assertTrue(all(p["prov"] is None for p in t["picks"]))


class TeamTests(unittest.TestCase):
    def test_team_table_is_complete_and_unique(self):
        self.assertEqual(len(bd.TEAMS), 30)
        ids = sorted(v[0] for v in bd.TEAMS.values())
        self.assertEqual(ids, list(range(1610612737, 1610612767)))  # the 30 consecutive official NBA team IDs
        for abbr, (tid, name, color) in bd.TEAMS.items():
            self.assertRegex(abbr, r"^[A-Z]{3}$")
            self.assertRegex(color, r"^#[0-9A-Fa-f]{6}$")
            self.assertTrue(name)

    def test_known_abbreviations_match_official_ids(self):
        # spot checks against the TEAM_ID / team pairs in data/nba_data.csv
        for abbr, tid in {"CLE": 1610612739, "GSW": 1610612744, "NOP": 1610612740, "BKN": 1610612751,
                          "SAS": 1610612759, "OKC": 1610612760, "CHA": 1610612766}.items():
            self.assertEqual(bd.TEAMS[abbr][0], tid)

    def test_with_team_attaches_most_recent_team(self):
        df = pd.DataFrame({"player": ["A", "A", "B"], "team": ["BOS", "NYK", None],
                           "game_date": ["2026-11-01", "2026-11-09", "2026-11-02"]})
        out = bd._with_team([{"player": "A"}, {"player": "B"}], df)
        self.assertEqual(out[0]["team"], "NYK")
        self.assertIsNone(out[1]["team"])

    def test_traded_player_gets_latest_team_even_when_file_order_is_not_chronological(self):
        # regression: picks_history.csv is not in date order; Kelly Oubre Jr. (PHI -> IND) showed his old team
        df = pd.DataFrame({"player": ["T", "T", "T"], "team": ["IND", "PHI", "PHI"],
                           "game_date": ["2026-11-10", "2026-11-02", "2026-11-04"]})   # newest row is first in the file
        self.assertEqual(bd._with_team([{"player": "T"}], df)[0]["team"], "IND")

    def test_trade_does_not_rewrite_history(self):
        # each pick keeps the team it was made for: a traded player's record is split by pick-time team
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            rows = pd.read_csv(root / "output" / "picks_history.csv")
            alpha_rows = rows[rows["player"] == "Alpha One"]
            rows.loc[alpha_rows.index[-1], "team"] = "IND"       # Alpha One's 11-02 pick is for his new team
            # an older pick appended at the END of the file (late backfill): file order != date order
            rows = pd.concat([rows, pd.DataFrame([_pick("2026-10-30", "Alpha One", "AST", "WIN", price=-110, line=5.5)])],
                             ignore_index=True)
            rows.to_csv(root / "output" / "picks_history.csv", index=False)
            season = bd.build_payload(root, now=NOW)["scopes"]["season"]
        teams = {t["team"]: t for t in season["teams_all"]}
        self.assertEqual((teams["IND"]["w"], teams["IND"]["l"]), (1, 0))   # the pick made for IND counts for IND
        self.assertEqual((teams["CLE"]["w"], teams["CLE"]["l"]), (2, 0))   # the two earlier picks still count for CLE
        alpha = next(p for p in season["players"] if p["player"] == "Alpha One")
        self.assertEqual(alpha["team"], "IND")                             # the player row shows the latest team by date
        self.assertEqual((alpha["w"], alpha["l"]), (3, 0))                 # and the player's full record stays together

    def test_payload_has_team_breakdowns_and_slate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            payload = bd.build_payload(root, now=NOW)
        self.assertEqual(set(payload["teams"]), set(bd.TEAMS))
        self.assertEqual(payload["teams"]["CLE"]["id"], 1610612739)
        season = payload["scopes"]["season"]
        teams = {t["team"]: t for t in season["teams_all"]}
        self.assertEqual((teams["CLE"]["w"], teams["CLE"]["l"]), (2, 0))
        self.assertEqual((teams["DEN"]["w"], teams["DEN"]["l"]), (0, 1))
        self.assertEqual((teams["BKN"]["w"], teams["BKN"]["l"]), (1, 0))
        opps = {t["opponent"]: t for t in season["opps"]}
        self.assertEqual((opps["DAL"]["w"], opps["DAL"]["l"]), (2, 0))  # opponent breakdown is a separate groupby
        self.assertEqual((opps["NOP"]["w"], opps["NOP"]["l"]), (0, 1))
        self.assertTrue(all(t["team"] for t in season["teams_all"]))   # no null-team bucket
        self.assertFalse(any(w.startswith("unknown team abbreviations") for w in payload["meta"]["warnings"]))
        games = payload["today"]["games"]
        self.assertEqual(len(games), 1)  # both card picks are CLE vs DAL
        self.assertEqual((games[0]["a"], games[0]["b"], games[0]["n"], games[0]["pending"]), ("CLE", "DAL", 2, 2))
        self.assertEqual(games[0]["tip"], "2026-11-05T00:10:00Z")

    def test_unknown_team_abbreviation_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            rows = pd.read_csv(root / "output" / "picks_history.csv")
            rows.loc[0, "team"] = "XYZ"
            rows.to_csv(root / "output" / "picks_history.csv", index=False)
            payload = bd.build_payload(root, now=NOW)
        self.assertTrue(any("XYZ" in w and "no logo" in w for w in payload["meta"]["warnings"]))
        self.assertEqual(len(payload["teams"]), 30)  # the dashboard still builds; XYZ gets a text badge

    def test_provisional_results_flow_into_the_slate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            _write_csv(root / "output" / "clv_picks.csv", [
                {"game_date": "2026-11-04", "player": "Echo Five", "stat": "ast", "line": 6.5, "direction": "UNDER", "result": "WIN", "actual": 4.0},
                {"game_date": "2026-11-04", "player": "Evil </script><b>x", "stat": "ast", "line": 2.5, "direction": "UNDER", "result": "LOSS", "actual": 3.0}])
            g = bd.build_payload(root, now=NOW)["today"]["games"][0]
        self.assertEqual((g["pw"], g["pl"], g["pending"], g["w"], g["l"]), (1, 1, 0, 0, 0))


class AssetContractTests(unittest.TestCase):
    """The page's only permitted external dependency is the NBA logo CDN, requested without a referrer,
    and only as the fallback for a team whose logo is not embedded."""

    ASSETS = Path(bd.ASSET_DIR)

    def _source(self):
        return {p.name: p.read_text(encoding="utf-8") for p in self.ASSETS.iterdir() if p.suffix in (".js", ".css", ".html")}

    def test_only_the_nba_logo_cdn_is_referenced(self):
        import re
        hosts = set()
        for name, text in self._source().items():
            hosts |= {h.lower() for h in re.findall(r"https?://([A-Za-z0-9.-]+)", text)}
        self.assertTrue(hosts <= {"cdn.nba.com", "www.w3.org"}, hosts)
        self.assertIn("cdn.nba.com", hosts)

    def test_logo_requests_send_no_referrer_and_have_a_text_fallback(self):
        app = self._source()["app.js"]
        self.assertIn("https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg", app)
        self.assertIn('referrerpolicy="no-referrer"', app)
        self.assertIn("nologo", app)                       # error handler flips to the abbreviation badge
        self.assertIn(".tlogo.nologo b", self._source()["style.css"])

    def test_embedded_logos_are_drawn_from_one_generated_stylesheet(self):
        app = self._source()["app.js"]
        self.assertIn("D.logos", app)
        self.assertIn(".tlogo.e-${a} i{background-image:url(", app)    # data URI stored once per team, not once per image
        self.assertIn("/^data:image\\/svg\\+xml;base64,[A-Za-z0-9+/=]+$/", app)  # only a plain base64 SVG data URI is ever injected
        self.assertIn(".tlogo i {", self._source()["style.css"])

    def test_logo_class_does_not_collide_with_the_decision_timeline(self):
        # regression: the logo box was also called .tl, so every logo inherited the timeline's padding/border
        # and the 84-entry decision log was squeezed into a 26px square
        import re
        css, app = self._source()["style.css"], self._source()["app.js"]
        self.assertEqual(len(re.findall(r"^\.tl \{", css, re.M)), 1)       # the decision-log timeline, defined once
        self.assertEqual(len(re.findall(r"^\.tlogo \{", css, re.M)), 1)    # the team-logo box, defined once
        self.assertIn('class="tl" id="dc-list"', app)
        self.assertIn('return `<span class="tlogo', app)                  # the logo helper emits only the logo class
        self.assertEqual(app.count('class="tl"'), 1)                       # ...and the timeline is the only user of .tl

    def test_page_says_how_old_it_is_and_flags_a_missed_daily_rebuild(self):
        # a static page cannot refresh itself: it must show its age and shout once it is older than a day
        app = self._source()["app.js"]
        self.assertIn("const STALE_HOURS = 26;", app)
        self.assertIn("ageH >= STALE_HOURS", app)
        self.assertIn('<div class="banner bad"><b>Out of date</b>', app)
        self.assertIn("ageTxt", app.split("$('#top').innerHTML", 1)[1].split("</div></div>", 1)[0])   # the header shows the age too
        self.assertIn(".banner.bad", self._source()["style.css"])

    def test_view_is_shown_before_it_renders(self):
        # regression: charts mounted inside a display:none view measured width 0 and stayed blank
        app = self._source()["app.js"]
        self.assertLess(app.index("el.classList.add('on');"), app.index("v.render(el)"))


class RenderTests(unittest.TestCase):
    def test_render_escapes_script_close_and_replaces_all_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            payload = bd.build_payload(root, now=NOW)
            html = bd.render_html(payload)
            out = root / "output" / "dashboard_pro.html"
            bd.write_atomic(out, html)
            self.assertTrue(out.exists())
            self.assertFalse(out.with_suffix(".html.tmp").exists())
        for token in ("/*__CSS__*/", "/*__DATA__*/", "/*__CHARTS__*/", "/*__APP__*/"):
            self.assertNotIn(token, html)
        # the hostile player name must never be able to terminate the data <script>
        data_block = html.split("window.__DASH__ = ", 1)[1].split(";</script>", 1)[0]
        self.assertNotIn("</", data_block)
        self.assertIn("<\\/script>", data_block)
        import re
        # no bare NaN value (a plain substring test would trip on base64 text such as an embedded wallpaper)
        self.assertIsNone(re.search(r"[:,\[]NaN[,\]}]", data_block))
        # the embedded JSON round-trips
        parsed = json.loads(data_block.replace("<\\/", "</"))
        self.assertEqual(parsed["scopes"]["season"]["summary"]["w"], 3)

    def test_main_survives_a_cp1252_console(self):
        # under Task Scheduler stdout is redirected (cp1252): a non-ASCII name in a warning must not crash the
        # run after the page was written
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            rows = pd.read_csv(root / "output" / "picks_history.csv")
            rows.loc[0, "team"] = "ĆĆĆ"            # not representable in cp1252
            rows.to_csv(root / "output" / "picks_history.csv", index=False, encoding="utf-8")
            sink = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
            with contextlib.redirect_stdout(sink):
                code = bd.main(["--root", str(root)])
            sink.flush()
            printed = sink.buffer.getvalue().decode("cp1252")
            self.assertEqual(code, 0)
            self.assertTrue((root / "output" / "dashboard_pro.html").exists())
        self.assertIn("\\u0106", printed)                          # spelled out, not a UnicodeEncodeError

    def test_main_writes_only_the_html(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            before = set(_hash_tree(root))
            self.assertEqual(bd.main(["--root", str(root)]), 0)
            after = _hash_tree(root)
            self.assertEqual(set(after) - before, {"output/dashboard_pro.html"})


GOOD_SVG = (b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 10 10">'
            b'<defs><linearGradient id="g"><stop offset="0" stop-color="#fff"/></linearGradient></defs>'
            b'<path id="a" d="M0 0h10v10z" fill="url(#g)"/><use xlink:href="#a"/></svg>')


def _svg(inner: str) -> bytes:
    return f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 10 10">{inner}</svg>'.encode()


class LogoEmbedTests(unittest.TestCase):
    """Team logos are committed files that get inlined; only plain, inert SVG may get in."""

    def test_clean_svg_with_internal_references_is_accepted(self):
        bd.validate_logo_svg(GOOD_SVG)
        # whitespace and quotes around a #fragment reference are still internal
        bd.validate_logo_svg(_svg('<path fill="url( #g )"/><path fill="url(\'#g\')"/><style>.a{fill:url( "#g" )}</style>'))

    def test_active_or_external_content_is_rejected(self):
        hostile = {
            "script": _svg("<script>alert(1)</script>"),
            "event attribute": _svg('<path d="M0 0" onload="alert(1)"/>'),
            "foreignObject": _svg("<foreignObject><div/></foreignObject>"),
            "external href": _svg('<use xlink:href="https://evil.example/x.svg#a"/>'),
            "plain href": _svg('<use href="logo.svg#a"/>'),
            "embedded image": _svg('<image href="#a"/>'),
            "external url()": _svg('<path fill="url(https://evil.example/g)"/>'),
            "style import": _svg("<style>@import url(https://evil.example/x.css);</style>"),
            "style url": _svg("<style>.a{fill:url(https://evil.example/g)}</style>"),
            "javascript uri": _svg('<path fill="javascript:alert(1)"/>'),
            "animate": _svg('<a><animate attributeName="href" values="javascript:alert(1)"/></a>'),
            "doctype": b'<!DOCTYPE svg [<!ENTITY x "y">]>' + _svg(""),
        }
        for label, raw in hostile.items():
            with self.subTest(label), self.assertRaises(ValueError):
                bd.validate_logo_svg(raw)

    def test_malformed_wrong_root_empty_and_oversized_are_rejected(self):
        for label, raw in {"malformed": b"<svg><path></svg>", "not svg": b"<html><body/></html>", "empty": b"",
                           "oversized": GOOD_SVG + b" " * bd.MAX_LOGO_BYTES}.items():
            with self.subTest(label), self.assertRaises(ValueError):
                bd.validate_logo_svg(raw)

    def test_load_logos_embeds_valid_files_and_reports_the_rest(self):
        import base64
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "ATL.svg").write_bytes(GOOD_SVG)
            (d / "BOS.svg").write_bytes(_svg("<script/>"))      # rejected, reported
            (d / "XYZ.svg").write_bytes(GOOD_SVG)               # not a team, ignored and reported
            (d / "notes.txt").write_text("not an svg")          # not even looked at
            logos, problems = bd.load_logos(d)
        self.assertEqual(list(logos), ["ATL"])
        prefix = "data:image/svg+xml;base64,"
        self.assertTrue(logos["ATL"].startswith(prefix))
        self.assertEqual(base64.b64decode(logos["ATL"][len(prefix):]), GOOD_SVG)   # embedded byte for byte
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("BOS.svg" in p and "rejected" in p for p in problems))
        self.assertTrue(any("XYZ.svg" in p and "not a known team" in p for p in problems))

    def test_missing_logo_directory_embeds_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(bd.load_logos(Path(td) / "nope"), ({}, []))

    def test_payload_says_loudly_which_logos_are_not_embedded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "proj"
            logos = Path(td) / "logos"
            logos.mkdir()
            make_project(root)
            (logos / "CLE.svg").write_bytes(GOOD_SVG)
            payload = bd.build_payload(root, now=NOW, logo_dir=logos)
        self.assertEqual(list(payload["logos"]), ["CLE"])
        self.assertEqual(payload["meta"]["logos_embedded"], 1)
        msg = [w for w in payload["meta"]["warnings"] if "not embedded" in w]
        self.assertEqual(len(msg), 1)
        self.assertIn("29 of 30", msg[0])
        self.assertNotIn("CLE", msg[0].split(":", 1)[1])         # the embedded team is not listed as missing
        self.assertIn("DAL", msg[0])

    def test_full_set_embeds_without_a_warning_and_survives_rendering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "proj"
            logos = Path(td) / "logos"
            logos.mkdir()
            make_project(root)
            for abbr in bd.TEAMS:
                (logos / f"{abbr}.svg").write_bytes(GOOD_SVG)
            payload = bd.build_payload(root, now=NOW, logo_dir=logos)
            html = bd.render_html(payload)
        self.assertEqual(set(payload["logos"]), set(bd.TEAMS))
        self.assertEqual(payload["meta"]["logos_embedded"], 30)
        self.assertFalse(any("not embedded" in w for w in payload["meta"]["warnings"]))
        data_block = html.split("window.__DASH__ = ", 1)[1].split(";</script>", 1)[0]
        parsed = json.loads(data_block.replace("<\\/", "</"))
        self.assertEqual(parsed["logos"]["CLE"], payload["logos"]["CLE"])

    def test_all_30_committed_logo_files_are_present_and_valid(self):
        # the offline dashboard needs every team: one <ABBR>.svg each, all passing the builder's own checks
        logos, problems = bd.load_logos()
        self.assertEqual(problems, [])
        self.assertEqual(set(logos), set(bd.TEAMS))
        for abbr, uri in logos.items():
            with self.subTest(abbr):
                self.assertTrue(uri.startswith("data:image/svg+xml;base64,"))


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16


class WallpaperTests(unittest.TestCase):
    """The optional page wallpaper is embedded byte for byte, and only if it is really an image."""

    def _load(self, name, data):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / name).write_bytes(data)
            return bd.load_wallpaper(d)

    def test_no_wallpaper_file_is_fine_and_silent(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(bd.load_wallpaper(Path(td)), (None, None))

    def test_valid_images_are_embedded_byte_for_byte_with_the_mime_from_their_magic_bytes(self):
        import base64
        for name, data, mime in (("wallpaper.jpg", JPEG_BYTES, "image/jpeg"), ("wallpaper.png", PNG_BYTES, "image/png"),
                                 ("wallpaper.webp", WEBP_BYTES, "image/webp")):
            with self.subTest(name):
                uri, problem = self._load(name, data)
                self.assertIsNone(problem)
                prefix = f"data:{mime};base64,"
                self.assertTrue(uri.startswith(prefix))
                self.assertEqual(base64.b64decode(uri[len(prefix):]), data)

    def test_a_file_that_is_not_an_image_is_rejected_whatever_its_extension(self):
        uri, problem = self._load("wallpaper.jpg", b"<html><script>alert(1)</script></html>")
        self.assertIsNone(uri)
        self.assertIn("not a JPEG, PNG or WebP", problem)
        self.assertIn("no wallpaper", problem)

    def test_empty_and_oversized_files_are_rejected(self):
        for label, data in {"empty": b"", "oversized": JPEG_BYTES + b"\x00" * bd.MAX_WALLPAPER_BYTES}.items():
            with self.subTest(label):
                uri, problem = self._load("wallpaper.jpg", data)
                self.assertIsNone(uri)
                self.assertIn("outside 1..", problem)

    def test_payload_carries_the_wallpaper_and_a_bad_one_becomes_a_warning(self):
        with tempfile.TemporaryDirectory() as td:
            root, assets = Path(td) / "proj", Path(td) / "assets"
            assets.mkdir()
            make_project(root)
            (assets / "wallpaper.jpg").write_bytes(JPEG_BYTES)
            good = bd.build_payload(root, now=NOW, wallpaper_dir=assets)
            (assets / "wallpaper.jpg").write_bytes(b"not an image")
            bad = bd.build_payload(root, now=NOW, wallpaper_dir=assets)
            none = bd.build_payload(root, now=NOW, wallpaper_dir=Path(td) / "empty")
        self.assertTrue(good["wallpaper"].startswith("data:image/jpeg;base64,"))
        self.assertFalse(any("wallpaper" in w for w in good["meta"]["warnings"]))
        self.assertIsNone(bad["wallpaper"])
        self.assertTrue(any("wallpaper" in w and "not a JPEG" in w for w in bad["meta"]["warnings"]))   # loud, not silent
        self.assertIsNone(none["wallpaper"])
        self.assertFalse(any("wallpaper" in w for w in none["meta"]["warnings"]))                       # optional: absence is fine

    def test_embedded_wallpaper_survives_rendering_and_adds_no_external_host(self):
        with tempfile.TemporaryDirectory() as td:
            root, assets = Path(td) / "proj", Path(td) / "assets"
            assets.mkdir()
            make_project(root)
            (assets / "wallpaper.png").write_bytes(PNG_BYTES)
            payload = bd.build_payload(root, now=NOW, wallpaper_dir=assets)
            html = bd.render_html(payload)
        data_block = html.split("window.__DASH__ = ", 1)[1].split(";</script>", 1)[0]
        self.assertEqual(json.loads(data_block.replace("<\\/", "</"))["wallpaper"], payload["wallpaper"])
        body = html.split("<body>", 1)[1].split("<script>", 1)[0]
        self.assertNotIn("<img", body)                                      # the page markup loads nothing: a CSS layer + data URI
        self.assertNotIn("http", body)

    def test_the_committed_wallpaper_is_valid(self):
        uri, problem = bd.load_wallpaper()                                  # the real assets folder (the file is optional)
        self.assertIsNone(problem)
        if uri is not None:
            self.assertRegex(uri, r"^data:image/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$")

    def test_front_end_only_injects_a_plain_image_data_uri_and_leaves_the_page_unchanged_when_off(self):
        import re
        src = {p.name: p.read_text(encoding="utf-8") for p in Path(bd.ASSET_DIR).iterdir() if p.suffix in (".js", ".css", ".html")}
        app, css = src["app.js"], src["style.css"]
        self.assertIn(r"/^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/", app)   # nothing else is ever injected
        self.assertIn('class="bgfx" aria-hidden="true"><div class="wall"></div>', src["template.html"])
        self.assertIn('aria-pressed="${wallOn}"', app)
        # every panel override lives behind [data-wall="on"], so switching it off restores the original look exactly
        block = css.split("optional wallpaper", 1)[1].split("/* ---------- layout", 1)[0]
        for line in block.splitlines():
            if "--panel:" in line or "--panel-hi:" in line:
                self.assertIn('[data-wall="on"]', line, line)
        self.assertEqual(len(re.findall(r"^\.bgfx \.wall \{", css, re.M)), 1)


class FailureTests(unittest.TestCase):
    def test_missing_picks_history_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                bd.build_payload(Path(td), now=NOW)

    def test_picks_missing_required_columns_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            _write_csv(Path(td) / "output" / "picks_history.csv", [{"game_date": "2026-11-01", "player": "x"}])
            with self.assertRaises(SystemExit):
                bd.build_payload(Path(td), now=NOW)

    def test_unusable_pick_rows_are_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_project(root)
            rows = pd.read_csv(root / "output" / "picks_history.csv")
            rows.loc[len(rows)] = {"result": "LOSS"}  # a row with nothing but a result
            rows.to_csv(root / "output" / "picks_history.csv", index=False)
            payload = bd.build_payload(root, now=NOW)
            self.assertEqual(payload["scopes"]["season"]["summary"]["l"], 1)  # unchanged by the junk row
            self.assertIn("unusable_picks", [f["kind"] for f in payload["health"]["flags"]])


if __name__ == "__main__":
    unittest.main()
