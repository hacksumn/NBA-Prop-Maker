from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"
ARCHIVE_DIR = OUTPUT_DIR / "archive"

TODAY_PICKS = OUTPUT_DIR / "picks_latest.csv"
TODAY_PREDICTIONS = OUTPUT_DIR / "predictions_latest.csv"
PICKS_HISTORY = OUTPUT_DIR / "picks_history.csv"
BETSLIPS_HISTORY = OUTPUT_DIR / "betslips_history.csv"
BETSLIPS_LATEST = OUTPUT_DIR / "betslips_latest.csv"
BACKTEST_SUMMARY = OUTPUT_DIR / "backtest_daily_summary.csv"
NBA_DATA = DATA_DIR / "nba_data.csv"
ABSENCES = DATA_DIR / "detected_absences.csv"
PLAYER_PROFILES = DATA_DIR / "player_profiles.csv"
TRAINING_RESULTS = MODELS_DIR / "training_results.json"
RESULTS_JSON = MODELS_DIR / "results.json"
FEATURE_IMPORTANCE = MODELS_DIR / "feature_importance_advanced.json"
TRAINING_EDGE = MODELS_DIR / "training_edge_analysis.json"
DASHBOARD_HTML = OUTPUT_DIR / "dashboard_latest.html"

COMMANDS = {
    "Run Daily Update": [sys.executable, "run_daily.py"],
    "Generate Picks": [sys.executable, "nba_props.py", "predict"],
    "Retrain Models": [sys.executable, "nba_props.py", "train"],
    "Advanced Retrain": [sys.executable, "train_advanced_models.py"],
    "Backtest": [sys.executable, "nba_props.py", "backtest"],
}

CARD_COLORS = ["#d6604d", "#2a9d8f", "#264653", "#e9c46a", "#577590", "#8f5d5d"]
STAT_OPTIONS = {
    "Points": "pts",
    "Rebounds": "trb",
    "Assists": "ast",
    "Steals": "stl",
    "Blocks": "blk",
    "Turnovers": "tov",
    "Minutes": "mp",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Vegas Intelligence Console",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --cream: #f4efe6;
            --paper: #fffaf2;
            --ink: #182127;
            --muted: #69757d;
            --accent: #d6604d;
            --accent-dark: #9d3f31;
            --gold: #d4a94b;
            --line: rgba(24, 33, 39, 0.10);
            --shadow: 0 18px 44px rgba(38, 70, 83, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(212, 169, 75, 0.22), transparent 26%),
                radial-gradient(circle at top left, rgba(214, 96, 77, 0.14), transparent 24%),
                linear-gradient(180deg, #fbf6ee 0%, #f4efe6 48%, #f1eadf 100%);
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, .stTabs [data-baseweb="tab"] {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.45);
            border-radius: 999px;
            border: 1px solid rgba(24, 33, 39, 0.08);
            padding: 0.55rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #d6604d, #f0b45f);
            color: white;
            border-color: transparent;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,250,242,0.88), rgba(255,250,242,0.70));
            border: 1px solid rgba(24, 33, 39, 0.08);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 0.9rem 1rem;
        }

        .hero-panel, .glass-panel {
            background: linear-gradient(180deg, rgba(255,250,242,0.84), rgba(255,250,242,0.72));
            border: 1px solid rgba(24, 33, 39, 0.08);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.15rem 1.2rem;
        }

        .hero-kicker {
            color: var(--accent-dark);
            font-size: 0.8rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .hero-title {
            font-size: 2.65rem;
            line-height: 0.98;
            margin: 0 0 0.35rem 0;
            font-weight: 700;
        }

        .hero-subtitle, .inline-note {
            color: var(--muted);
        }

        .inline-note {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.82rem;
        }

        .pick-card {
            background: linear-gradient(140deg, rgba(38,70,83,0.98), rgba(52,87,96,0.96));
            color: #fef8ef;
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 156px;
            border: 1px solid rgba(255,255,255,0.07);
            box-shadow: 0 18px 36px rgba(38, 70, 83, 0.18);
        }

        .pick-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.55rem;
        }

        .pick-team, .pick-metric-label {
            font-size: 0.72rem;
            opacity: 0.7;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .pick-player { font-size: 1.24rem; font-weight: 700; margin: 0; }

        .pick-line, .status-pill {
            display: inline-block;
            border-radius: 999px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.76rem;
        }

        .pick-line {
            margin-top: 0.28rem;
            padding: 0.2rem 0.55rem;
            background: rgba(255,255,255,0.12);
        }

        .status-pill {
            padding: 0.18rem 0.65rem;
            background: rgba(212,169,75,0.16);
        }

        .pick-body {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }

        .pick-metric-value { font-size: 1.08rem; font-weight: 600; margin-top: 0.15rem; }

        .sidebar-section {
            background: rgba(255,250,242,0.70);
            border-radius: 16px;
            padding: 0.8rem 0.85rem;
            border: 1px solid rgba(24, 33, 39, 0.08);
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_pct(value: Optional[float], scale: float = 1.0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * scale:.1f}%"


def format_num(value: Optional[float], digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def format_int(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def human_stamp(path: Path) -> str:
    if not path.exists():
        return "missing"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime_ns: int, parse_dates: Tuple[str, ...]) -> pd.DataFrame:
    kwargs = {}
    if parse_dates:
        kwargs["parse_dates"] = list(parse_dates)
    return pd.read_csv(path_str, low_memory=False, **kwargs)


def load_csv(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not safe_exists(path):
        return pd.DataFrame()
    return _read_csv_cached(str(path), path.stat().st_mtime_ns, tuple(parse_dates or ()))


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime_ns: int) -> Dict:
    with open(path_str, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> Dict:
    if not safe_exists(path):
        return {}
    return _read_json_cached(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def list_archives(root_str: str, mtime_ns: int) -> List[Dict[str, str]]:
    root = Path(root_str)
    entries: List[Dict[str, str]] = []
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            entries.append(
                {
                    "date_folder": path.parent.name,
                    "name": path.name,
                    "suffix": path.suffix.lower(),
                    "path": str(path),
                }
            )
    return entries


def get_archive_files() -> pd.DataFrame:
    if not ARCHIVE_DIR.exists():
        return pd.DataFrame(columns=["date_folder", "name", "suffix", "path"])
    newest_touch = max((p.stat().st_mtime_ns for p in ARCHIVE_DIR.rglob("*")), default=0)
    return pd.DataFrame(list_archives(str(ARCHIVE_DIR), newest_touch))


def load_log_files() -> List[Path]:
    if not LOGS_DIR.exists():
        return []
    return sorted([path for path in LOGS_DIR.iterdir() if path.is_file()], key=lambda path: path.stat().st_mtime, reverse=True)


def get_latest_df(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = load_csv(path, parse_dates=parse_dates)
    return df.copy() if not df.empty else pd.DataFrame()


def prep_pick_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    numeric_cols = ["line", "prediction", "edge", "confidence", "actual", "mp_predicted", "calibrated_conf", "dir_prob", "meta_prob", "market_edge", "rank_score"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def prep_nba_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    numeric_cols = ["mp", "pts", "trb", "ast", "stl", "blk", "tov", "fg", "fga", "3p", "3pa", "ft", "fta", "plus_minus"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def graded_picks(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "result" not in history.columns:
        return pd.DataFrame()
    return history[history["result"].isin(["WIN", "LOSS", "PUSH"])].copy()


def decisions_only(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    return history[history["result"].isin(["WIN", "LOSS"])].copy()


def plot_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(l=16, r=16, t=40, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,242,0.30)",
        font=dict(family="Space Grotesk, sans-serif", color="#182127"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(24, 33, 39, 0.10)")
    return fig


def build_overview_metrics(
    today: pd.DataFrame,
    history: pd.DataFrame,
    backtest: pd.DataFrame,
    betslips: pd.DataFrame,
) -> Dict[str, str]:
    graded = decisions_only(history)
    wins = int((graded["result"] == "WIN").sum()) if not graded.empty else 0
    losses = int((graded["result"] == "LOSS").sum()) if not graded.empty else 0
    avg_conf = today["confidence"].mean() if "confidence" in today.columns and not today.empty else None
    latest_top3 = None
    if not backtest.empty and "top_3_hit_rate" in backtest.columns:
        series = backtest["top_3_hit_rate"].dropna()
        if not series.empty:
            latest_top3 = series.iloc[-1]
    slip_wins = int((betslips.get("result") == "WIN").sum()) if not betslips.empty and "result" in betslips.columns else 0
    slip_losses = int((betslips.get("result") == "LOSS").sum()) if not betslips.empty and "result" in betslips.columns else 0
    return {
        "today_picks": format_int(len(today)),
        "avg_conf": format_num(avg_conf, 1),
        "all_time_wr": format_pct((wins / (wins + losses) * 100) if (wins + losses) else None),
        "backtest_top3": format_pct(latest_top3 * 100 if latest_top3 is not None else None),
        "slip_record": f"{slip_wins}-{slip_losses}",
    }


def render_pick_cards(today: pd.DataFrame) -> None:
    if today.empty:
        st.info("No picks are available yet.")
        return

    cols = st.columns(3)
    for idx, row in today.reset_index(drop=True).iterrows():
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="pick-card">
                    <div class="pick-header">
                        <div>
                            <div class="pick-team">{row.get('team', 'n/a')}</div>
                            <p class="pick-player">{row.get('player', 'Unknown')}</p>
                            <span class="pick-line">{row.get('prop', '')} {row.get('direction', '')} {row.get('line', '')}</span>
                        </div>
                        <span class="status-pill">{format_num(row.get('confidence'), 1)} conf</span>
                    </div>
                    <div class="pick-body">
                        <div>
                            <div class="pick-metric-label">Projection</div>
                            <div class="pick-metric-value">{format_num(row.get('prediction'), 1)}</div>
                        </div>
                        <div>
                            <div class="pick-metric-label">Model Edge</div>
                            <div class="pick-metric-value">{format_num(row.get('edge'), 1)}</div>
                        </div>
                        <div>
                            <div class="pick-metric-label">Dir Prob</div>
                            <div class="pick-metric-value">{format_num(row.get('dir_prob'), 1)}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_streaming_command(label: str, command: List[str]) -> None:
    terminal = st.empty()
    status = st.status(f"{label} is running", expanded=True)
    lines: List[str] = [f"$ {' '.join(command)}", ""]
    terminal.code("\n".join(lines), language="bash")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        lines.append(raw_line.rstrip("\n"))
        terminal.code("\n".join(lines[-450:]), language="bash")

    rc = process.wait()
    finished = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["terminal_output"] = "\n".join(lines[-2000:])
    st.session_state["terminal_rc"] = rc
    st.session_state["terminal_label"] = label
    st.session_state["terminal_finished"] = finished

    if rc == 0:
        status.update(label=f"{label} finished successfully", state="complete", expanded=False)
        st.cache_data.clear()
        time.sleep(0.5)
        st.rerun()
    else:
        status.update(label=f"{label} failed with exit code {rc}", state="error", expanded=True)


def render_terminal_controls() -> None:
    st.markdown("### Command Console")
    st.caption("Run the real project scripts from inside the UI. Output streams into the attached terminal below.")

    button_cols = st.columns(len(COMMANDS))
    for idx, (label, command) in enumerate(COMMANDS.items()):
        with button_cols[idx]:
            if st.button(label, key=f"cmd_{idx}", width="stretch"):
                run_streaming_command(label, command)

    terminal_output = st.session_state.get("terminal_output", "")
    terminal_rc = st.session_state.get("terminal_rc")
    terminal_label = st.session_state.get("terminal_label", "No command run yet")
    terminal_finished = st.session_state.get("terminal_finished", "")
    header = terminal_label if not terminal_finished else f"{terminal_label} · {terminal_finished}"
    if terminal_rc is not None:
        header = f"{header} · exit {terminal_rc}"
    st.caption(header)
    st.code(terminal_output or "# Terminal output will appear here", language="bash")


def render_sidebar(health_rows: List[Tuple[str, Path]]) -> None:
    with st.sidebar:
        st.markdown("## Vegas Console")
        st.caption("Operational dashboard for the `Fresh_Start_NBA` pipeline.")

        if st.button("Refresh Data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### Project Health")
        for label, path in health_rows:
            st.markdown(f"`{label}`  \n{human_stamp(path)} · {human_size(path)}")
        st.markdown("</div>", unsafe_allow_html=True)

        logs = load_log_files()
        if logs:
            latest_log = logs[0]
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("#### Latest Log")
            st.markdown(f"[{latest_log.name}]({latest_log.as_posix()})")
            st.caption(human_stamp(latest_log))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### Launch Notes")
        st.caption("Use `Generate Picks` for the live card, `Run Daily Update` for ingestion + grading, and retrains only when you want new model artifacts.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_hero(today: pd.DataFrame) -> None:
    latest_run = human_stamp(TODAY_PICKS)
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">Vegas Intelligence Console</div>
            <div class="hero-title">Fresh Start NBA Mission Control</div>
            <p class="hero-subtitle">
                One local control room for picks, box scores, model diagnostics, archive review, and pipeline execution.
                Latest picks snapshot: <span class="inline-note">{latest_run}</span>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if today.empty:
        st.warning("`output/picks_latest.csv` is missing or empty. Run `Generate Picks` from the console.")


def render_mission_control(
    today: pd.DataFrame,
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    betslips: pd.DataFrame,
) -> None:
    render_hero(today)

    metrics = build_overview_metrics(today, history, backtest, betslips)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Today Picks", metrics["today_picks"])
    m2.metric("Avg Confidence", metrics["avg_conf"])
    m3.metric("All-Time Win Rate", metrics["all_time_wr"])
    m4.metric("Backtest Top-3", metrics["backtest_top3"])
    m5.metric("Betslip Record", metrics["slip_record"])

    left, right = st.columns([1.4, 1.0], gap="large")
    with left:
        st.markdown("### Live Card")
        render_pick_cards(today)
        if not today.empty:
            view_cols = ["player", "team", "prop", "direction", "line", "prediction", "edge", "confidence", "dir_prob", "meta_prob", "market_edge", "mp_predicted"]
            available = [col for col in view_cols if col in today.columns]
            st.dataframe(today[available], width="stretch", height=330)

            if len(today) > 1:
                chart = px.scatter(
                    today,
                    x="confidence",
                    y="edge",
                    color="prop",
                    size="dir_prob" if "dir_prob" in today.columns else None,
                    hover_name="player",
                    hover_data=["direction", "line", "prediction"],
                    color_discrete_sequence=CARD_COLORS,
                )
                chart.update_traces(marker=dict(line=dict(color="white", width=1.2), opacity=0.88))
                plot_template(chart)
                chart.update_layout(title="Confidence vs Model Edge")
                st.plotly_chart(chart, width="stretch")

    with right:
        render_terminal_controls()
        st.markdown("### File Health")
        log_files = load_log_files()
        health_df = pd.DataFrame(
            [
                {"artifact": "picks_latest.csv", "updated": human_stamp(TODAY_PICKS), "size": human_size(TODAY_PICKS)},
                {"artifact": "predictions_latest.csv", "updated": human_stamp(TODAY_PREDICTIONS), "size": human_size(TODAY_PREDICTIONS)},
                {"artifact": "dashboard_latest.html", "updated": human_stamp(DASHBOARD_HTML), "size": human_size(DASHBOARD_HTML)},
                {"artifact": "picks_history.csv", "updated": human_stamp(PICKS_HISTORY), "size": human_size(PICKS_HISTORY)},
                {"artifact": "run log", "updated": human_stamp(log_files[0]) if log_files else "missing", "size": human_size(log_files[0]) if log_files else "missing"},
            ]
        )
        st.dataframe(health_df, width="stretch", hide_index=True)

    with st.expander("Latest HTML Dashboard", expanded=False):
        if DASHBOARD_HTML.exists():
            st.html(DASHBOARD_HTML, unsafe_allow_javascript=True)
        else:
            st.info("`output/dashboard_latest.html` does not exist yet.")


def render_picks_board(today: pd.DataFrame, betslips_latest: pd.DataFrame) -> None:
    st.markdown("### Picks Board")
    if today.empty:
        st.info("No live picks available.")
        return

    filter_cols = st.columns(4)
    props = sorted(today["prop"].dropna().unique().tolist()) if "prop" in today.columns else []
    teams = sorted(today["team"].dropna().unique().tolist()) if "team" in today.columns else []
    directions = sorted(today["direction"].dropna().unique().tolist()) if "direction" in today.columns else []

    selected_props = filter_cols[0].multiselect("Prop", props, default=props)
    selected_teams = filter_cols[1].multiselect("Team", teams, default=teams)
    selected_dirs = filter_cols[2].multiselect("Direction", directions, default=directions)
    min_conf = filter_cols[3].slider("Minimum Confidence", 0, 100, 60)

    filtered = today.copy()
    if selected_props:
        filtered = filtered[filtered["prop"].isin(selected_props)]
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]
    if selected_dirs:
        filtered = filtered[filtered["direction"].isin(selected_dirs)]
    if "confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["confidence"], errors="coerce") >= min_conf]

    render_pick_cards(filtered)

    st.markdown("#### Current Picks Table")
    st.dataframe(filtered, width="stretch", height=360)

    if not filtered.empty and "prop" in filtered.columns:
        left, right = st.columns(2)
        with left:
            mix = px.bar(
                filtered.groupby(["prop", "direction"], dropna=False).size().reset_index(name="count"),
                x="prop",
                y="count",
                color="direction",
                barmode="group",
                color_discrete_sequence=[CARD_COLORS[0], CARD_COLORS[2]],
                title="Card Mix by Market",
            )
            plot_template(mix)
            st.plotly_chart(mix, width="stretch")
        with right:
            conf = px.bar(
                filtered.sort_values("confidence", ascending=True),
                x="confidence",
                y="player",
                color="prop",
                orientation="h",
                color_discrete_sequence=CARD_COLORS,
                title="Confidence Ladder",
            )
            plot_template(conf)
            st.plotly_chart(conf, width="stretch")

    st.markdown("#### Suggested Betslips")
    if betslips_latest.empty:
        st.info("No latest betslips file found.")
    else:
        st.dataframe(betslips_latest, width="stretch", height=220)


def render_performance_lab(history: pd.DataFrame, backtest: pd.DataFrame, betslips_history: pd.DataFrame) -> None:
    st.markdown("### Performance Lab")
    graded = graded_picks(history)
    decisions = decisions_only(history)

    if decisions.empty:
        st.info("There are not enough graded picks yet to build performance charts.")
        return

    daily = (
        graded.groupby("game_date", dropna=False)["result"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("game_date")
    )
    for col in ["WIN", "LOSS", "PUSH"]:
        if col not in daily.columns:
            daily[col] = 0
    daily["decisions"] = daily["WIN"] + daily["LOSS"]
    daily["hit_rate"] = daily["WIN"] / daily["decisions"].replace(0, pd.NA)
    daily["cum_wins"] = daily["WIN"].cumsum()
    daily["cum_losses"] = daily["LOSS"].cumsum()

    bucketed = decisions.copy()
    bucketed["conf_bucket"] = pd.cut(
        pd.to_numeric(bucketed["confidence"], errors="coerce"),
        bins=[0, 60, 70, 80, 101],
        labels=["<60", "60-69", "70-79", "80+"],
        include_lowest=True,
    )
    bucket_summary = (
        bucketed.groupby("conf_bucket", dropna=False, observed=False)["result"]
        .apply(lambda s: (s == "WIN").mean() if len(s) else None)
        .reset_index(name="hit_rate")
    )

    prop_summary = (
        decisions.groupby("prop", dropna=False)["result"]
        .apply(lambda s: (s == "WIN").mean() if len(s) else None)
        .reset_index(name="hit_rate")
        .sort_values("hit_rate", ascending=False)
    )
    prop_summary["games"] = decisions.groupby("prop").size().values

    top = st.columns(4)
    total_wins = int((decisions["result"] == "WIN").sum())
    total_losses = int((decisions["result"] == "LOSS").sum())
    total_pushes = int((graded["result"] == "PUSH").sum())
    top[0].metric("Decisions", format_int(len(decisions)))
    top[1].metric("Record", f"{total_wins}-{total_losses}")
    top[2].metric("Pushes", format_int(total_pushes))
    top[3].metric("Decision Win Rate", format_pct(total_wins / (total_wins + total_losses) * 100 if total_wins + total_losses else None))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["game_date"], y=daily["cum_wins"], mode="lines", name="Wins", line=dict(color="#2a9d8f", width=3)))
        fig.add_trace(go.Scatter(x=daily["game_date"], y=daily["cum_losses"], mode="lines", name="Losses", line=dict(color="#d6604d", width=3)))
        plot_template(fig)
        fig.update_layout(title="Cumulative Wins vs Losses")
        st.plotly_chart(fig, width="stretch")
    with chart_right:
        hit = px.bar(
            prop_summary,
            x="prop",
            y="hit_rate",
            color="games",
            color_continuous_scale="Brwnyl",
            title="Win Rate by Prop Market",
            text="games",
        )
        plot_template(hit)
        hit.update_yaxes(tickformat=".0%")
        st.plotly_chart(hit, width="stretch")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        conf = px.bar(
            bucket_summary,
            x="conf_bucket",
            y="hit_rate",
            color="conf_bucket",
            color_discrete_sequence=CARD_COLORS,
            title="Win Rate by Confidence Bucket",
        )
        plot_template(conf)
        conf.update_yaxes(tickformat=".0%")
        st.plotly_chart(conf, width="stretch")
    with lower_right:
        rate = px.line(daily, x="game_date", y="hit_rate", markers=True, title="Daily Hit Rate")
        rate.update_traces(line=dict(color="#264653", width=3), marker=dict(color="#d4a94b", size=8))
        plot_template(rate)
        rate.update_yaxes(tickformat=".0%")
        st.plotly_chart(rate, width="stretch")

    if not backtest.empty:
        st.markdown("#### Backtest Ladder")
        bt_cols = [col for col in ["top_1_hit_rate", "top_3_hit_rate", "top_5_hit_rate", "top_10_hit_rate"] if col in backtest.columns]
        if bt_cols:
            bt_long = backtest[["game_date", *bt_cols]].melt("game_date", var_name="ladder", value_name="hit_rate").dropna()
            bt_chart = px.line(
                bt_long,
                x="game_date",
                y="hit_rate",
                color="ladder",
                markers=True,
                color_discrete_sequence=CARD_COLORS,
                title="Historical Daily-Card Backtest",
            )
            plot_template(bt_chart)
            bt_chart.update_yaxes(tickformat=".0%")
            st.plotly_chart(bt_chart, width="stretch")

    if not betslips_history.empty and "result" in betslips_history.columns:
        st.markdown("#### Betslip Outcomes")
        slip_summary = (
            betslips_history[betslips_history["result"].isin(["WIN", "LOSS", "PUSH"])]
            .groupby(["slip_type", "result"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        if not slip_summary.empty:
            st.dataframe(slip_summary, width="stretch", hide_index=True)


def render_player_profile_card(profile_row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="hero-kicker">{profile_row.get('team_abbr', 'TEAM')}</div>
            <div style="font-size:1.6rem;font-weight:700;margin-bottom:0.2rem;">{profile_row.get('player_name', 'Unknown')}</div>
            <div class="pick-body" style="color:#182127;">
                <div><div class="pick-metric-label">Role</div><div class="pick-metric-value">{profile_row.get('role_tier', 'n/a')}</div></div>
                <div><div class="pick-metric-label">USG%</div><div class="pick-metric-value">{format_pct(profile_row.get('usg_pct'), 100)}</div></div>
                <div><div class="pick-metric-label">TS%</div><div class="pick-metric-value">{format_pct(profile_row.get('ts_pct'), 100)}</div></div>
                <div><div class="pick-metric-label">PTS/G</div><div class="pick-metric-value">{format_num(profile_row.get('pts_pg'), 1)}</div></div>
                <div><div class="pick-metric-label">REB/G</div><div class="pick-metric-value">{format_num(profile_row.get('reb_pg'), 1)}</div></div>
                <div><div class="pick-metric-label">AST/G</div><div class="pick-metric-value">{format_num(profile_row.get('ast_pg'), 1)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_explorer(nba_data: pd.DataFrame, profiles: pd.DataFrame) -> None:
    st.markdown("### Data Explorer")
    if nba_data.empty:
        st.info("`data/nba_data.csv` is missing or unreadable.")
        return

    teams = sorted(nba_data["team"].dropna().astype(str).unique().tolist()) if "team" in nba_data.columns else []
    players = sorted(nba_data["player"].dropna().astype(str).unique().tolist()) if "player" in nba_data.columns else []

    control_left, control_mid, control_right = st.columns([1.2, 1.2, 1.0])
    selected_player = control_left.selectbox("Player", players, index=0 if players else None)
    selected_stat = control_mid.selectbox("Trend Stat", list(STAT_OPTIONS.keys()), index=0)
    selected_team = control_right.selectbox("Team Filter", ["All Teams", *teams], index=0)

    player_df = nba_data[nba_data["player"] == selected_player].sort_values("game_date")
    if selected_team != "All Teams":
        player_df = player_df[player_df["team"] == selected_team]

    if not profiles.empty and "player_name" in profiles.columns:
        profile_match = profiles[profiles["player_name"] == selected_player]
        if not profile_match.empty:
            render_player_profile_card(profile_match.iloc[0])

    stat_col = STAT_OPTIONS[selected_stat]
    trend = player_df.tail(20).copy()
    trend["rolling_5"] = trend[stat_col].rolling(5, min_periods=1).mean()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(x=trend["game_date"], y=trend[stat_col], name=selected_stat, marker_color="#d6604d", opacity=0.75))
    trend_fig.add_trace(go.Scatter(x=trend["game_date"], y=trend["rolling_5"], mode="lines+markers", name="Rolling 5", line=dict(color="#264653", width=3)))
    plot_template(trend_fig)
    trend_fig.update_layout(title=f"{selected_player} · Last 20 Games")
    st.plotly_chart(trend_fig, width="stretch")

    recent_cols = ["game_date", "team", "opp", "matchup", "mp", "pts", "trb", "ast", "stl", "blk", "tov", "plus_minus", "result"]
    available_recent = [col for col in recent_cols if col in player_df.columns]
    st.markdown("#### Recent Box Scores")
    st.dataframe(player_df.sort_values("game_date", ascending=False)[available_recent].head(20), width="stretch", height=360)

    st.markdown("#### Game Box Score Lens")
    latest_dates = sorted(nba_data["game_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist(), reverse=True)
    game_cols = st.columns([1.0, 1.6])
    selected_date = game_cols[0].selectbox("Game Date", latest_dates, index=0 if latest_dates else None)
    daily = nba_data[nba_data["game_date"].dt.strftime("%Y-%m-%d") == selected_date].copy()
    matchups = sorted(daily["matchup"].dropna().unique().tolist()) if "matchup" in daily.columns else []
    selected_matchup = game_cols[1].selectbox("Matchup", matchups, index=0 if matchups else None)
    box = daily[daily["matchup"] == selected_matchup].copy() if selected_matchup else pd.DataFrame()
    if not box.empty:
        team_totals = box.groupby("team")[["pts", "trb", "ast", "stl", "blk", "tov"]].sum().reset_index().sort_values("pts", ascending=False)
        team_cols = st.columns(2)
        with team_cols[0]:
            st.dataframe(team_totals, width="stretch", hide_index=True)
        with team_cols[1]:
            show_cols = ["player", "team", "mp", "pts", "trb", "ast", "stl", "blk", "tov", "fg", "fga", "3p", "3pa", "ft", "fta", "plus_minus"]
            available_show = [col for col in show_cols if col in box.columns]
            st.dataframe(box.sort_values(["team", "pts"], ascending=[True, False])[available_show], width="stretch", height=420)


def build_market_table(training_results: Dict) -> pd.DataFrame:
    rows = []
    for market, payload in training_results.items():
        real_summary = payload.get("real_line_summary", {})
        direction_summary = {row.get("label"): row for row in real_summary.get("direction_summary", [])}
        market_summary = payload.get("market_edge_summary", {})
        meta_summary = payload.get("meta_summary", {})
        rows.append(
            {
                "market": market.upper(),
                "mae": payload.get("avg_mae"),
                "rmse": payload.get("avg_rmse"),
                "real_hit_rate": real_summary.get("hit_rate"),
                "under_hit_rate": direction_summary.get("under_edges", {}).get("hit_rate"),
                "over_hit_rate": direction_summary.get("over_edges", {}).get("hit_rate"),
                "market_edge_hit_rate": market_summary.get("hit_rate"),
                "meta_auc": meta_summary.get("auc"),
                "top_meta_hit_rate": meta_summary.get("top_prob_hit_rate"),
                "real_lines": payload.get("n_real_lines"),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(feature_importance: Dict, market: str) -> pd.DataFrame:
    payload = feature_importance.get(market.lower(), {})
    if not payload:
        return pd.DataFrame(columns=["feature", "importance"])
    rows = [{"feature": feature, "importance": value} for feature, value in payload.items()]
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def render_model_room(training_results: Dict, results_json: Dict, feature_importance: Dict, training_edge: Dict) -> None:
    st.markdown("### Model Room")
    if not training_results:
        st.info("Training diagnostics are not available yet.")
        return

    market_df = build_market_table(training_results)
    top_left, top_right = st.columns(2)
    with top_left:
        acc = px.bar(
            market_df.sort_values("real_hit_rate", ascending=False),
            x="market",
            y="real_hit_rate",
            color="market",
            color_discrete_sequence=CARD_COLORS,
            title="Real-Line Hit Rate by Market",
        )
        plot_template(acc)
        acc.update_yaxes(tickformat=".0%")
        st.plotly_chart(acc, width="stretch")
    with top_right:
        meta = px.bar(
            market_df.sort_values("meta_auc", ascending=False),
            x="market",
            y="meta_auc",
            color="market",
            color_discrete_sequence=CARD_COLORS,
            title="Meta AUC by Market",
        )
        plot_template(meta)
        st.plotly_chart(meta, width="stretch")

    st.dataframe(
        market_df.style.format(
            {
                "mae": "{:.2f}",
                "rmse": "{:.2f}",
                "real_hit_rate": "{:.1%}",
                "under_hit_rate": "{:.1%}",
                "over_hit_rate": "{:.1%}",
                "market_edge_hit_rate": "{:.1%}",
                "meta_auc": "{:.3f}",
                "top_meta_hit_rate": "{:.1%}",
            }
        ),
        width="stretch",
        height=340,
    )

    markets = market_df["market"].tolist()
    selected_market = st.selectbox("Inspect Market", markets, index=0 if markets else None)
    payload = training_results.get(selected_market.lower(), {}) if selected_market else {}
    real_summary = payload.get("real_line_summary", {})
    market_summary = payload.get("market_edge_summary", {})
    meta_summary = payload.get("meta_summary", {})

    info_cols = st.columns(4)
    info_cols[0].metric("MAE", format_num(payload.get("avg_mae"), 2))
    info_cols[1].metric("Real-Line Hit", format_pct(real_summary.get("hit_rate") * 100 if real_summary.get("hit_rate") is not None else None))
    info_cols[2].metric("Market-Edge Hit", format_pct(market_summary.get("hit_rate") * 100 if market_summary.get("hit_rate") is not None else None))
    info_cols[3].metric("Meta AUC", format_num(meta_summary.get("auc"), 3))

    mid_left, mid_right = st.columns(2)
    with mid_left:
        bucket_df = pd.DataFrame(real_summary.get("bucket_summary", []))
        if not bucket_df.empty:
            bucket_chart = px.bar(
                bucket_df,
                x="label",
                y="hit_rate",
                color="avg_edge",
                color_continuous_scale="Earth",
                title=f"{selected_market} Edge Buckets",
                text="n",
            )
            plot_template(bucket_chart)
            bucket_chart.update_yaxes(tickformat=".0%")
            st.plotly_chart(bucket_chart, width="stretch")
    with mid_right:
        fi = feature_importance_table(feature_importance, selected_market or "")
        if not fi.empty:
            fi_chart = px.bar(
                fi.head(15).sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Tealrose",
                title=f"{selected_market} Top Features",
            )
            plot_template(fi_chart)
            st.plotly_chart(fi_chart, width="stretch")

    st.markdown("#### Direction Summary")
    st.dataframe(pd.DataFrame(real_summary.get("direction_summary", [])), width="stretch", hide_index=True)

    if results_json:
        st.markdown("#### Legacy Results Snapshot")
        rows = [{"market": key.upper(), **value} for key, value in results_json.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    cross_target = training_edge.get("cross_target_direction_summary")
    if cross_target:
        st.markdown("#### Cross-Target Direction Summary")
        st.dataframe(pd.DataFrame(cross_target), width="stretch", hide_index=True)


def render_archive_and_logs(absences: pd.DataFrame, archive_files: pd.DataFrame) -> None:
    st.markdown("### Archives, Logs, and Roster Intel")

    top_left, top_right = st.columns([1.1, 1.1], gap="large")
    with top_left:
        st.markdown("#### Current Absences")
        if absences.empty:
            st.info("`data/detected_absences.csv` is missing or empty.")
        else:
            st.dataframe(absences, width="stretch", height=320)

    with top_right:
        st.markdown("#### Log Browser")
        logs = load_log_files()
        if logs:
            selected_log = st.selectbox("Log File", logs, format_func=lambda path: path.name)
            tail_lines = st.slider("Tail Lines", 50, 600, 180, 10)
            text = selected_log.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(text[-tail_lines:]), language="bash")
        else:
            st.info("No log files found.")

    st.markdown("#### Archive Browser")
    if archive_files.empty:
        st.info("No archived outputs found yet.")
        return

    date_options = sorted(archive_files["date_folder"].dropna().unique().tolist(), reverse=True)
    selector_cols = st.columns([1.0, 1.0, 1.8])
    selected_date = selector_cols[0].selectbox("Archive Date", date_options, index=0 if date_options else None)
    day_files = archive_files[archive_files["date_folder"] == selected_date].copy()
    suffix_options = sorted(day_files["suffix"].dropna().unique().tolist())
    selected_suffix = selector_cols[1].selectbox("File Type", suffix_options, index=0 if suffix_options else None)
    candidate_files = day_files[day_files["suffix"] == selected_suffix].copy()
    file_paths = candidate_files["path"].tolist()
    selected_file = selector_cols[2].selectbox("File", file_paths, format_func=lambda item: Path(item).name, index=0 if file_paths else None)
    if not selected_file:
        return

    target = Path(selected_file)
    st.caption(f"{target.name} · {human_stamp(target)} · {human_size(target)}")
    if target.suffix.lower() == ".csv":
        preview = pd.read_csv(target)
        st.dataframe(preview, width="stretch", height=520)
        st.download_button("Download CSV", data=target.read_bytes(), file_name=target.name, mime="text/csv", width="stretch")
    elif target.suffix.lower() == ".html":
        st.html(target, unsafe_allow_javascript=True)
    else:
        st.code(target.read_text(encoding="utf-8", errors="replace"), language="text")


def main() -> None:
    configure_page()
    today = get_latest_df(TODAY_PICKS)
    predictions = get_latest_df(TODAY_PREDICTIONS)
    history = prep_pick_history(get_latest_df(PICKS_HISTORY, parse_dates=["game_date"]))
    betslips_history = get_latest_df(BETSLIPS_HISTORY)
    betslips_latest = get_latest_df(BETSLIPS_LATEST)
    backtest = get_latest_df(BACKTEST_SUMMARY, parse_dates=["game_date"])
    nba_data = prep_nba_data(get_latest_df(NBA_DATA, parse_dates=["game_date"]))
    absences = get_latest_df(ABSENCES)
    profiles = get_latest_df(PLAYER_PROFILES)
    training_results = load_json(TRAINING_RESULTS)
    results_json = load_json(RESULTS_JSON)
    feature_importance = load_json(FEATURE_IMPORTANCE)
    training_edge = load_json(TRAINING_EDGE)
    archive_files = get_archive_files()

    render_sidebar(
        [
            ("Live picks", TODAY_PICKS),
            ("Predictions", TODAY_PREDICTIONS),
            ("History", PICKS_HISTORY),
            ("NBA data", NBA_DATA),
            ("Training results", TRAINING_RESULTS),
        ]
    )

    tabs = st.tabs(["Mission Control", "Picks Board", "Performance Lab", "Data Explorer", "Model Room", "Archives & Logs"])
    with tabs[0]:
        render_mission_control(today, history, predictions, backtest, betslips_history)
    with tabs[1]:
        render_picks_board(today, betslips_latest)
    with tabs[2]:
        render_performance_lab(history, backtest, betslips_history)
    with tabs[3]:
        render_data_explorer(nba_data, profiles)
    with tabs[4]:
        render_model_room(training_results, results_json, feature_importance, training_edge)
    with tabs[5]:
        render_archive_and_logs(absences, archive_files)


if __name__ == "__main__":
    main()
