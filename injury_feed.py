"""Shared live injury feed helpers for official NBA + ESPN status ingestion."""

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

import requests


_OFFICIAL_NBA_TEAM_TO_ABBR = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}

_INJURY_BUCKET_ORDER = ['out', 'doubtful', 'questionable', 'probable', 'day_to_day', 'available']


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for cross-source player matching."""
    import unicodedata

    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return re.sub(r"[^a-z ]", '', name.lower()).strip()


def _classify_injury_bucket(item: dict) -> Optional[str]:
    """Map ESPN injury payloads to a coarse pregame availability bucket."""
    status = str(item.get('status', '') or '').strip().lower()
    type_desc = str(item.get('type', {}).get('description', '') or '').strip().lower()
    short_comment = str(item.get('shortComment', '') or '').strip().lower()
    long_comment = str(item.get('longComment', '') or '').strip().lower()
    text = ' '.join([type_desc, short_comment, long_comment])

    if status == 'out' or type_desc == 'out':
        return 'out'
    if 'doubtful' in text:
        return 'doubtful'
    if 'questionable' in text or 'game-time decision' in text or 'gameday decision' in text:
        return 'questionable'
    if 'probable' in text or 'expected to play' in text or 'will play' in text:
        return 'probable'
    if status == 'day-to-day' or 'day-to-day' in text or 'day to day' in text:
        return 'day_to_day'
    return None


def _new_injury_entry() -> Dict[str, object]:
    entry = {bucket: [] for bucket in _INJURY_BUCKET_ORDER}
    entry['status_map'] = {}
    entry['not_yet_submitted'] = False
    entry['report_label'] = ''
    entry['report_url'] = ''
    return entry


def _ensure_injury_entry(injuries: dict, team_abbr: str) -> Dict[str, object]:
    if team_abbr not in injuries:
        injuries[team_abbr] = _new_injury_entry()
    return injuries[team_abbr]


def _set_injury_status(entry: Dict[str, object],
                       player_norm: str,
                       bucket: str,
                       *,
                       status: str = '',
                       comment: str = '',
                       source: str = '',
                       overwrite: bool = False) -> None:
    if bucket not in _INJURY_BUCKET_ORDER:
        return
    current = entry.get('status_map', {}).get(player_norm)
    if current and not overwrite:
        return
    for existing_bucket in _INJURY_BUCKET_ORDER:
        if player_norm in entry.get(existing_bucket, []):
            entry[existing_bucket] = [x for x in entry[existing_bucket] if x != player_norm]
    entry.setdefault(bucket, []).append(player_norm)
    entry.setdefault('status_map', {})[player_norm] = {
        'bucket': bucket,
        'status': status,
        'comment': comment,
        'source': source,
    }


def _parse_official_report_name(tokens: List[str]) -> Optional[str]:
    for i in range(len(tokens) - 1, -1, -1):
        if ',' not in tokens[i]:
            continue
        start = i
        if tokens[i] in {'Jr.,', 'Sr.,', 'II,', 'III,', 'IV,', 'V,'} and start > 0:
            start -= 1
        while start > 0 and tokens[start - 1].endswith('-'):
            start -= 1
        raw = ' '.join(tokens[start:]).replace('- ', '-').strip()
        if ',' not in raw:
            continue
        last, first = raw.split(',', 1)
        full_name = f"{first.strip()} {last.strip()}".strip()
        return full_name or None
    return None


def fetch_official_nba_injury_data() -> dict:
    """Fetch the latest official NBA injury report PDF for current-day statuses."""
    try:
        from bs4 import BeautifulSoup
        from pypdf import PdfReader
    except Exception:
        return {}

    page_url = "https://official.nba.com/nba-injury-report-2025-26-season/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        page = requests.get(page_url, headers=headers, timeout=45)
        if page.status_code != 200:
            return {}
        soup = BeautifulSoup(page.text, 'html.parser')
        report_links = []
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            text = anchor.get_text(' ', strip=True)
            if 'ak-static.cms.nba.com' in href and 'report' in text.lower():
                report_links.append((text, href))
        if not report_links:
            return {}

        report_label, report_url = report_links[-1]
        pdf_resp = requests.get(report_url, headers=headers, timeout=45)
        if pdf_resp.status_code != 200 or not pdf_resp.content:
            return {}

        reader = PdfReader(io.BytesIO(pdf_resp.content))
        flat_text = re.sub(r'\s+', ' ', ' '.join(page.extract_text() or '' for page in reader.pages)).strip()
        if not flat_text:
            return {}

        team_names = sorted(_OFFICIAL_NBA_TEAM_TO_ABBR, key=len, reverse=True)
        team_alt = '|'.join(re.escape(name) for name in team_names)
        team_pat = re.compile(rf'({team_alt})(.*?)(?=({team_alt})|\d{{1,2}}:\d{{2}} \(ET\)|$)')
        status_bucket_map = {
            'Out': 'out',
            'Doubtful': 'doubtful',
            'Questionable': 'questionable',
            'Probable': 'probable',
            'Available': 'available',
        }
        injuries = {}
        for team_name, segment, _ in team_pat.findall(flat_text):
            team_abbr = _OFFICIAL_NBA_TEAM_TO_ABBR.get(team_name)
            if not team_abbr:
                continue
            entry = _ensure_injury_entry(injuries, team_abbr)
            entry['report_label'] = report_label
            entry['report_url'] = report_url
            pending_segment = 'NOT YET SUBMITTED' in segment.upper()
            found_status = False

            tokens = segment.split()
            for idx, token in enumerate(tokens):
                if token not in status_bucket_map:
                    continue
                full_name = _parse_official_report_name(tokens[max(0, idx - 5):idx])
                if not full_name:
                    continue
                player_norm = _normalize_name(full_name)
                _set_injury_status(
                    entry,
                    player_norm,
                    status_bucket_map[token],
                    status=token,
                    comment='NBA official injury report',
                    source='nba_official',
                    overwrite=True,
                )
                found_status = True

            if found_status:
                entry['not_yet_submitted'] = False
            elif pending_segment and not entry.get('status_map'):
                entry['not_yet_submitted'] = True

        return {
            team: entry for team, entry in injuries.items()
            if entry.get('not_yet_submitted') or entry.get('status_map')
        }
    except Exception as exc:
        print(f"  Official NBA injury report unavailable: {exc}")
        return {}


def _fetch_espn_injury_data() -> dict:
    """Fetch current ESPN injury data keyed by team abbreviation."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {}
        data = response.json()
        injuries = {}
        for team in data.get('injuries', []):
            for item in team.get('injuries', []):
                name = item.get('athlete', {}).get('displayName', '')
                abbr = (
                    item.get('athlete', {}).get('team', {}).get('abbreviation', '') or
                    team.get('abbreviation', '') or
                    ''
                )
                if not abbr or not name:
                    continue
                bucket = _classify_injury_bucket(item)
                if bucket is None:
                    continue

                team_entry = _ensure_injury_entry(injuries, abbr)
                player_norm = _normalize_name(name)
                _set_injury_status(
                    team_entry,
                    player_norm,
                    bucket,
                    status=item.get('status', ''),
                    comment=item.get('shortComment', '') or item.get('longComment', ''),
                    source='espn',
                    overwrite=True,
                )
        return injuries
    except Exception as exc:
        print(f"  Injury data unavailable: {exc}")
        return {}


def fetch_injury_data() -> dict:
    """Fetch combined injury/status data, preferring the official NBA report over ESPN."""
    official = fetch_official_nba_injury_data()
    espn = _fetch_espn_injury_data()

    injuries = {}
    for source_data in [official, espn]:
        for team_abbr, team_data in source_data.items():
            entry = _ensure_injury_entry(injuries, team_abbr)
            if team_data.get('not_yet_submitted'):
                entry['not_yet_submitted'] = True
            if team_data.get('report_label'):
                entry['report_label'] = team_data.get('report_label', '')
            if team_data.get('report_url'):
                entry['report_url'] = team_data.get('report_url', '')

    for team_abbr, team_data in official.items():
        entry = _ensure_injury_entry(injuries, team_abbr)
        for player_norm, detail in team_data.get('status_map', {}).items():
            _set_injury_status(
                entry,
                player_norm,
                detail.get('bucket', ''),
                status=detail.get('status', ''),
                comment=detail.get('comment', ''),
                source=detail.get('source', 'nba_official'),
                overwrite=True,
            )

    for team_abbr, team_data in espn.items():
        entry = _ensure_injury_entry(injuries, team_abbr)
        for player_norm, detail in team_data.get('status_map', {}).items():
            if player_norm in entry.get('status_map', {}):
                continue
            _set_injury_status(
                entry,
                player_norm,
                detail.get('bucket', ''),
                status=detail.get('status', ''),
                comment=detail.get('comment', ''),
                source=detail.get('source', 'espn'),
                overwrite=False,
            )

    return injuries
