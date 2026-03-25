# NBA Prop Model Data Fetcher

A free, automated data pipeline to replace paid Odds API subscriptions for NBA player prop modeling.

## What it does
1. **NBA Game Logs**: Pulls complete player game logs directly from NBA.com via `nba_api`. Includes every stat you need (pts, trb, ast, stl, blk, tov, mp, fga, fg, fg_pct, fta, ft, 3pa, 3p, 3p_pct, result, plus_minus) with home/away derived context.
2. **PrizePicks Prop Lines**: Scrapes today's lines directly from the PrizePicks API. It bypasses Cloudflare/PerimeterX bot detection by piggybacking on your existing Chrome browser profile.

---

## Setup Instructions

### 1. Install Requirements
You'll need Python 3 installed. Then install the required packages:
```bash
pip install pandas requests nba_api playwright
playwright install chromium
```

### 2. Configure PrizePicks Scraper (Important!)
PrizePicks blocks standard bot traffic. To bypass this, the scraper uses **your local Chrome/Chromium browser profile** because your browser has already passed their CAPTCHA checks.

1. Open Chrome on your computer and go to [app.prizepicks.com](https://app.prizepicks.com)
2. Open the `prizepicks_scraper.py` file in a text editor.
3. Look for `CHROME_PROFILE` at the top of the file and set it to your Chrome profile path.
   - **Windows:** `C:/Users/<YourUsername>/AppData/Local/Google/Chrome/User Data`
   - **macOS:** `/Users/<YourUsername>/Library/Application Support/Google/Chrome`
   - **Linux:** `/home/<YourUsername>/.config/google-chrome`

*(Alternatively, you can set the `PRIZEPICKS_CHROME_PROFILE` environment variable).*

---

## How to Use

Run the master script each morning before you run your model:

```bash
# Recommended daily run (Incremental update)
# Pulls last night's box scores and today's prop lines
python run_daily.py
```

### Other Commands
```bash
# If you ever need to re-download all 3 seasons from scratch
python run_daily.py --full-refresh

# If you only want to update the prop lines
python run_daily.py --lines-only

# If you only want to update the NBA game logs
python run_daily.py --logs-only
```

## Data Outputs
The script creates and maintains two CSV files that plug directly into your model:

1. `data/nba_data.csv` - The historical game logs.
2. `data/historical_lines.csv` - The daily prop lines.
