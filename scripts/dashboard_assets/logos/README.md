# Team logos for the Command Center dashboard

One file per team, named by its abbreviation in `TEAMS` (`scripts/build_dashboard.py`), e.g. `CLE.svg`.
`python scripts/build_dashboard.py` validates each file and inlines it into `output/dashboard_pro.html`,
so the dashboard needs no internet. A file that fails validation is listed in the page's warning banner.

- **Source:** the NBA's public logo CDN, `https://cdn.nba.com/logos/nba/<teamId>/global/L/logo.svg`
  (team IDs are in `TEAMS`). Downloaded 2026-11-15; official NBA team marks, used only for this local dashboard.
- **To refresh one:** replace its `.svg` and rerun `run_dashboard.bat`.
- `tests/test_build_dashboard.py` requires all 30 files to be present and valid.
