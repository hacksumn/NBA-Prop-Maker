from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent.resolve()
_DEFAULT_DATA_DIR = _HERE / "data"
_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_SPECIAL_FIRST_NAME_ALIASES = {
    "bub": {"carlton"},
    "carlton": {"bub"},
    "moe": {"moritz"},
    "moritz": {"moe"},
}


@dataclass(frozen=True)
class SeasonPlayerLookup:
    season: str
    alias_to_canonical: Dict[str, str]
    fuzzy_records: Tuple[Tuple[str, str, str], ...]
    has_roster: bool


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9 ]", " ", ascii_str.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def infer_nba_season(date_value) -> str:
    ts = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(ts):
        return ""
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _suffixless_key(name: str) -> str:
    tokens = normalize_name(name).split()
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _compact_initials_key(name: str) -> str:
    compact = normalize_name(name)
    previous = None
    while compact != previous:
        previous = compact
        compact = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", compact)
    return compact.strip()


def _current_nba_season() -> str:
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _roster_cache_path(season: str, data_dir: Path) -> Path:
    return data_dir / "reference" / "official_rosters" / f"official_roster_{season}.csv"


def _cache_is_fresh(path: Path, season: str, max_age_hours: int = 18) -> bool:
    if not path.exists():
        return False
    if season != _current_nba_season():
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= max_age_hours * 3600


def _fetch_roster_cache(season: str, cache_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    for team in nba_teams.get_teams():
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                roster = commonteamroster.CommonTeamRoster(
                    team_id=team["id"],
                    season=season,
                    timeout=30,
                ).get_data_frames()[0]
                subset = roster[["PLAYER", "PLAYER_ID"]].copy()
                subset["team_abbr"] = team["abbreviation"]
                rows.extend(
                    subset.rename(
                        columns={"PLAYER": "player_name", "PLAYER_ID": "player_id"}
                    ).to_dict("records")
                )
                break
            except Exception as exc:  # pragma: no cover - network retry path
                last_exc = exc
                time.sleep(0.75 * (attempt + 1))
        else:
            errors.append(f"{team['abbreviation']}: {last_exc}")

    if errors:
        raise RuntimeError(
            f"Could not build official roster cache for {season}. Failures: {', '.join(errors[:5])}"
        )

    roster_df = pd.DataFrame(rows).drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    roster_df.to_csv(cache_path, index=False)
    return roster_df


def _load_roster_cache(season: str, data_dir: Path) -> Tuple[pd.DataFrame, bool]:
    cache_path = _roster_cache_path(season, data_dir)
    if _cache_is_fresh(cache_path, season):
        return pd.read_csv(cache_path, low_memory=False), True

    try:
        return _fetch_roster_cache(season, cache_path), True
    except Exception as exc:
        if cache_path.exists():
            log.warning("Using stale official roster cache for %s after refresh failed: %s", season, exc)
            return pd.read_csv(cache_path, low_memory=False), True
        log.warning("Official roster fetch failed for %s: %s", season, exc)
        return pd.DataFrame(columns=["player_name", "player_id", "team_abbr"]), False


def _load_local_aliases(season: str, data_dir: Path) -> pd.DataFrame:
    nba_path = data_dir / "nba_data.csv"
    if not nba_path.exists():
        return pd.DataFrame(columns=["player_id", "player_name"])

    try:
        local = pd.read_csv(
            nba_path,
            usecols=["PLAYER_ID", "player", "season"],
            low_memory=False,
        )
    except ValueError:
        return pd.DataFrame(columns=["player_id", "player_name"])

    local = local[local["season"].astype(str) == season].copy()
    if local.empty:
        return pd.DataFrame(columns=["player_id", "player_name"])

    local["player_id"] = pd.to_numeric(local["PLAYER_ID"], errors="coerce").astype("Int64")
    local["player_name"] = local["player"].astype(str).str.strip()
    local = (
        local.dropna(subset=["player_id", "player_name"])
        .groupby(["player_id", "player_name"], as_index=False)
        .size()
        .sort_values(["player_id", "size", "player_name"], ascending=[True, False, True])
    )
    return local[["player_id", "player_name"]]


def _first_name_matches(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 3 and right.startswith(left):
        return True
    if len(right) >= 3 and left.startswith(right):
        return True
    if right in _SPECIAL_FIRST_NAME_ALIASES.get(left, set()):
        return True
    if left in _SPECIAL_FIRST_NAME_ALIASES.get(right, set()):
        return True
    return False


def _bucket_alias(alias_buckets: Dict[str, set], alias: str, canonical_name: str) -> None:
    for key in {normalize_name(alias), _suffixless_key(alias), _compact_initials_key(alias)}:
        if key:
            alias_buckets.setdefault(key, set()).add(canonical_name)


def _build_lookup(season: str, roster_df: pd.DataFrame, local_aliases: pd.DataFrame) -> SeasonPlayerLookup:
    records: Dict[object, Dict[str, object]] = {}

    if not roster_df.empty:
        roster_df = roster_df.copy()
        roster_df["player_id"] = pd.to_numeric(roster_df["player_id"], errors="coerce").astype("Int64")
        roster_df["player_name"] = roster_df["player_name"].astype(str).str.strip()
        for row in roster_df.itertuples(index=False):
            if pd.isna(row.player_id):
                continue
            records[int(row.player_id)] = {
                "canonical_name": row.player_name,
                "aliases": {row.player_name},
            }

    if not local_aliases.empty:
        for row in local_aliases.itertuples(index=False):
            if pd.isna(row.player_id):
                continue
            pid = int(row.player_id)
            record = records.setdefault(
                pid,
                {"canonical_name": row.player_name, "aliases": set()},
            )
            record["aliases"].add(row.player_name)

    alias_buckets: Dict[str, set] = {}
    fuzzy_records: List[Tuple[str, str, str]] = []

    for record in records.values():
        canonical_name = str(record["canonical_name"]).strip()
        aliases = {canonical_name, *{str(alias).strip() for alias in record["aliases"] if str(alias).strip()}}
        for alias in aliases:
            _bucket_alias(alias_buckets, alias, canonical_name)

        suffixless = _suffixless_key(canonical_name)
        parts = suffixless.split()
        if len(parts) >= 2:
            fuzzy_records.append((canonical_name, parts[0], parts[-1]))

    alias_to_canonical = {
        alias: next(iter(names))
        for alias, names in alias_buckets.items()
        if len(names) == 1
    }
    return SeasonPlayerLookup(
        season=season,
        alias_to_canonical=alias_to_canonical,
        fuzzy_records=tuple(sorted(set(fuzzy_records))),
        has_roster=not roster_df.empty,
    )


@lru_cache(maxsize=8)
def load_season_player_lookup(season: str, data_dir_value: str | None = None) -> SeasonPlayerLookup:
    data_dir = Path(data_dir_value) if data_dir_value else _DEFAULT_DATA_DIR
    roster_df, has_roster = _load_roster_cache(season, data_dir)
    local_aliases = _load_local_aliases(season, data_dir)
    lookup = _build_lookup(season, roster_df, local_aliases)
    if has_roster != lookup.has_roster:
        return SeasonPlayerLookup(
            season=lookup.season,
            alias_to_canonical=lookup.alias_to_canonical,
            fuzzy_records=lookup.fuzzy_records,
            has_roster=has_roster,
        )
    return lookup


def canonicalize_player_name(name: str, lookup: SeasonPlayerLookup) -> Optional[str]:
    for key in (normalize_name(name), _suffixless_key(name), _compact_initials_key(name)):
        if key and key in lookup.alias_to_canonical:
            return lookup.alias_to_canonical[key]

    parts = _suffixless_key(name).split()
    if len(parts) < 2:
        return None

    first_name = parts[0]
    last_name = parts[-1]
    candidates = {
        canonical_name
        for canonical_name, canonical_first, canonical_last in lookup.fuzzy_records
        if canonical_last == last_name and _first_name_matches(first_name, canonical_first)
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def sanitize_player_names(
    df: pd.DataFrame,
    *,
    player_col: str = "player",
    date_col: str = "game_date",
    player_norm_col: Optional[str] = "player_norm",
    data_dir: Optional[Path] = None,
    drop_unknown: bool = True,
    require_roster: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if df is None or len(df) == 0:
        return df.copy(), {
            "rows_in": 0,
            "rows_out": 0,
            "dropped_rows": 0,
            "canonicalized_rows": 0,
            "unknown_players": [],
            "seasons": [],
        }

    if player_col not in df.columns:
        raise KeyError(f"Missing required player column: {player_col}")
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    working = df.copy()
    working["_season_key"] = working[date_col].apply(infer_nba_season)

    cleaned_parts: List[pd.DataFrame] = []
    unknown_names: set[str] = set()
    canonicalized_rows = 0
    seasons_used: List[str] = []

    for season, season_df in working.groupby("_season_key", sort=True):
        season = str(season or "").strip()
        if not season:
            if drop_unknown:
                unknown_names.update(season_df[player_col].dropna().astype(str).tolist())
                continue
            cleaned_parts.append(season_df)
            continue

        lookup = load_season_player_lookup(season, str(data_dir))
        if require_roster and not lookup.has_roster:
            raise RuntimeError(f"Official NBA roster validation unavailable for season {season}")

        canonical_names = season_df[player_col].apply(lambda name: canonicalize_player_name(name, lookup))
        matched = season_df.copy()
        matched["_canonical_player"] = canonical_names

        unmatched = matched["_canonical_player"].isna()
        if unmatched.any():
            unknown_names.update(matched.loc[unmatched, player_col].dropna().astype(str).tolist())
            if drop_unknown:
                matched = matched.loc[~unmatched].copy()
            else:
                matched.loc[unmatched, "_canonical_player"] = matched.loc[unmatched, player_col]

        if matched.empty:
            continue

        changed = matched[player_col].astype(str) != matched["_canonical_player"].astype(str)
        canonicalized_rows += int(changed.sum())
        matched[player_col] = matched["_canonical_player"]
        if player_norm_col:
            matched[player_norm_col] = matched[player_col].apply(normalize_name)
        matched = matched.drop(columns=["_canonical_player"], errors="ignore")
        cleaned_parts.append(matched)
        seasons_used.append(season)

    if cleaned_parts:
        cleaned = pd.concat(cleaned_parts, ignore_index=False)
        cleaned = cleaned.drop(columns=["_season_key"], errors="ignore")
        cleaned = cleaned.reset_index(drop=True)
    else:
        cleaned = working.iloc[0:0].drop(columns=["_season_key"], errors="ignore").copy()

    summary = {
        "rows_in": int(len(df)),
        "rows_out": int(len(cleaned)),
        "dropped_rows": int(len(df) - len(cleaned)),
        "canonicalized_rows": int(canonicalized_rows),
        "unknown_players": sorted(unknown_names),
        "seasons": sorted(set(seasons_used)),
    }
    return cleaned, summary


def repair_historical_lines_file(path: Path | str, *, data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    cleaned, summary = sanitize_player_names(
        frame,
        player_col="player",
        date_col="game_date",
        player_norm_col="player_norm",
        data_dir=data_dir,
        drop_unknown=True,
        require_roster=False,
    )
    cleaned.to_csv(path, index=False)
    return cleaned, summary
