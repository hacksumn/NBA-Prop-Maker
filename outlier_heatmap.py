"""
outlier_heatmap.py — Find training data outliers
Outputs: outlier_heatmap.html (open in browser)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(r"C:\Users\jakep\Downloads\Fresh_Start_NBA\data\nba_data.csv")
OUT  = Path(r"C:\Users\jakep\Downloads\Fresh_Start_NBA\output\outlier_heatmap.html")

df = pd.read_csv(DATA, low_memory=False)
for col in ['mp','pts','trb','ast','stl','blk','tov']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

STATS = ['pts','trb','ast','stl','blk','tov']

# ── 1. Flag outlier categories ─────────────────────────────────────────────────
flags = pd.DataFrame(index=df.index)

# Garbage time / DNP-adjacent — played under 5 minutes
flags['low_minutes']   = df['mp'] < 5

# Extreme stat games (> 4 std from player mean)
for s in STATS:
    player_mean = df.groupby('player')[s].transform('mean')
    player_std  = df.groupby('player')[s].transform('std').fillna(1).clip(lower=0.5)
    flags[f'{s}_zscore'] = ((df[s] - player_mean) / player_std).abs()
    flags[f'{s}_outlier'] = flags[f'{s}_zscore'] > 4

# Very high minutes (OT games etc)
flags['high_minutes']  = df['mp'] > 45

# Zero-stat games with normal minutes (possible data error)
flags['zero_pts_normal_mp'] = (df['pts'] == 0) & (df['mp'] >= 15)

# Combine: any outlier flag
stat_outlier_cols = [f'{s}_outlier' for s in STATS]
flags['any_stat_outlier'] = flags[stat_outlier_cols].any(axis=1)
flags['any_flag']         = flags[['low_minutes','any_stat_outlier',
                                    'high_minutes','zero_pts_normal_mp']].any(axis=1)

df_flagged = df.copy()
df_flagged['flag_low_minutes']     = flags['low_minutes']
df_flagged['flag_high_minutes']    = flags['high_minutes']
df_flagged['flag_zero_pts']        = flags['zero_pts_normal_mp']
df_flagged['flag_stat_outlier']    = flags['any_stat_outlier']
for s in STATS:
    df_flagged[f'zscore_{s}']      = flags[f'{s}_zscore'].round(2)
df_flagged['any_flag']             = flags['any_flag']

# ── 2. Summary table: counts by flag type ──────────────────────────────────────
summary = {
    'Low Minutes (<5 mp)':          flags['low_minutes'].sum(),
    'High Minutes (>45 mp)':        flags['high_minutes'].sum(),
    'Zero PTS with >=15 mp':        flags['zero_pts_normal_mp'].sum(),
    'Stat Outlier (>4 std)':        flags['any_stat_outlier'].sum(),
    'Total Flagged Rows':           flags['any_flag'].sum(),
    'Total Rows':                   len(df),
}
pct_flagged = summary['Total Flagged Rows'] / summary['Total Rows'] * 100

# ── 3. Per-stat outlier breakdown ──────────────────────────────────────────────
stat_counts = {s: flags[f'{s}_outlier'].sum() for s in STATS}

# ── 4. Top outlier rows to display ─────────────────────────────────────────────
display_cols = ['player','game_date','mp','pts','trb','ast','stl','blk','tov',
                'flag_low_minutes','flag_high_minutes','flag_zero_pts','flag_stat_outlier'] + \
               [f'zscore_{s}' for s in STATS]

top_outliers = df_flagged[df_flagged['any_flag']][display_cols].copy()
top_outliers['game_date'] = top_outliers['game_date'].dt.strftime('%Y-%m-%d')
top_outliers = top_outliers.sort_values('zscore_pts', ascending=False).head(200)

# ── 5. Heatmap: outlier rate by player (top 50 most flagged) ──────────────────
player_flag_rate = (
    df_flagged.groupby('player')['any_flag'].agg(['sum','count'])
    .rename(columns={'sum':'flagged','count':'games'})
)
player_flag_rate['rate'] = (player_flag_rate['flagged'] / player_flag_rate['games'] * 100).round(1)
player_flag_rate = player_flag_rate[player_flag_rate['games'] >= 10].sort_values('flagged', ascending=False).head(50)

# ── 6. Build HTML ──────────────────────────────────────────────────────────────
def color_rate(val, max_val=50):
    """Red gradient for high flag rates."""
    pct = min(val / max_val, 1.0)
    r = int(255 * pct)
    g = int(255 * (1 - pct * 0.8))
    b = 80
    return f'background-color: rgb({r},{g},{b}); color: {"white" if pct > 0.5 else "black"}'

def flag_cell(val):
    if val:
        return '<td style="background:#e74c3c;color:white;text-align:center">YES</td>'
    return '<td style="text-align:center">-</td>'

def zscore_cell(val):
    if val >= 4:
        color = '#c0392b'
    elif val >= 3:
        color = '#e67e22'
    elif val >= 2:
        color = '#f1c40f'
    else:
        color = 'transparent'
    text_color = 'white' if val >= 3 else 'black'
    return f'<td style="background:{color};color:{text_color};text-align:center">{val:.1f}</td>'

# Stat outlier bar chart data
bar_labels = list(stat_counts.keys())
bar_values = list(stat_counts.values())
bar_colors = ['#e74c3c','#e67e22','#f39c12','#27ae60','#2980b9','#8e44ad']

# Player heatmap rows
player_rows = ""
for player, row in player_flag_rate.iterrows():
    pct = row['rate']
    max_val = 60
    intensity = min(pct / max_val, 1.0)
    r = int(200 * intensity + 55)
    g = int(200 * (1 - intensity * 0.8))
    b = 80
    bg = f'rgb({r},{g},{b})'
    tc = 'white' if intensity > 0.5 else 'black'
    player_rows += f"""
    <tr>
        <td>{player}</td>
        <td style="text-align:center">{int(row['games'])}</td>
        <td style="text-align:center">{int(row['flagged'])}</td>
        <td style="background:{bg};color:{tc};text-align:center;font-weight:bold">{pct}%</td>
    </tr>"""

# Outlier detail rows
detail_rows = ""
for _, row in top_outliers.iterrows():
    detail_rows += "<tr>"
    detail_rows += f"<td>{row['player']}</td>"
    detail_rows += f"<td>{row['game_date']}</td>"
    detail_rows += f"<td style='text-align:center'>{row['mp']:.1f}</td>"
    for s in ['pts','trb','ast','stl','blk','tov']:
        detail_rows += f"<td style='text-align:center'>{int(row[s])}</td>"
    detail_rows += flag_cell(row['flag_low_minutes'])
    detail_rows += flag_cell(row['flag_high_minutes'])
    detail_rows += flag_cell(row['flag_zero_pts'])
    detail_rows += flag_cell(row['flag_stat_outlier'])
    for s in STATS:
        detail_rows += zscore_cell(row[f'zscore_{s}'])
    detail_rows += "</tr>"

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NBA Training Data Outlier Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ color: #e74c3c; margin-bottom: 4px; }}
  h2 {{ color: #3498db; border-bottom: 1px solid #3498db; padding-bottom: 4px; margin-top: 30px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #0f3460; }}
  .card .num {{ font-size: 2em; font-weight: bold; color: #e74c3c; }}
  .card .label {{ font-size: 0.85em; color: #aaa; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; margin-top: 10px; }}
  th {{ background: #0f3460; color: #fff; padding: 8px 10px; text-align: left; position: sticky; top: 0; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #2a2a4a; }}
  tr:hover td {{ background: rgba(255,255,255,0.05); }}
  .bar-chart {{ display: flex; align-items: flex-end; gap: 16px; height: 180px; margin: 20px 0; padding: 10px; background: #16213e; border-radius: 8px; }}
  .bar-wrap {{ display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; }}
  .bar {{ width: 100%; border-radius: 4px 4px 0 0; transition: opacity 0.2s; min-height: 4px; }}
  .bar-label {{ font-size: 0.75em; margin-top: 4px; color: #aaa; }}
  .bar-val {{ font-size: 0.85em; font-weight: bold; margin-bottom: 4px; }}
  .scroll-wrap {{ overflow-x: auto; }}
  .note {{ background: #0f3460; border-left: 4px solid #e74c3c; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.88em; }}
</style>
</head>
<body>
<h1>NBA Training Data — Outlier Report</h1>
<p style="color:#aaa">Dataset: {summary['Total Rows']:,} rows &nbsp;|&nbsp; Flagged: <strong style="color:#e74c3c">{summary['Total Flagged Rows']:,} ({pct_flagged:.1f}%)</strong></p>

<h2>Summary</h2>
<div class="summary-grid">
  <div class="card"><div class="num">{summary['Low Minutes (<5 mp)']:,}</div><div class="label">Low Minutes &lt;5 mp<br>(garbage time / DNP)</div></div>
  <div class="card"><div class="num">{summary['High Minutes (>45 mp)']:,}</div><div class="label">High Minutes &gt;45 mp<br>(OT games)</div></div>
  <div class="card"><div class="num">{summary['Zero PTS with >=15 mp']:,}</div><div class="label">0 PTS with &ge;15 minutes<br>(possible data error)</div></div>
  <div class="card"><div class="num">{summary['Stat Outlier (>4 std)']:,}</div><div class="label">Stat Outlier &gt;4 std dev<br>(extreme games)</div></div>
  <div class="card"><div class="num">{summary['Total Flagged Rows']:,}</div><div class="label">Total Flagged Rows</div></div>
  <div class="card"><div class="num">{pct_flagged:.1f}%</div><div class="label">Pct of Dataset Flagged</div></div>
</div>

<div class="note">
  <strong>What to remove:</strong> Low-minutes rows (&lt;5 mp) pollute rolling averages with garbage time zeros.
  Stat outliers (&gt;4 std) skew EWMA features toward one-off career games.
  Zero-PTS with normal minutes may be data errors. High-minutes OT games are real but noisy.
</div>

<h2>Outliers by Stat</h2>
<div class="bar-chart">
{"".join(f'''<div class="bar-wrap">
  <div class="bar-val" style="color:{bar_colors[i]}">{bar_values[i]}</div>
  <div class="bar" style="height:{int(bar_values[i]/max(bar_values)*150)}px;background:{bar_colors[i]}"></div>
  <div class="bar-label">{bar_labels[i].upper()}</div>
</div>''' for i, _ in enumerate(bar_labels))}
</div>

<h2>Most Flagged Players (>=10 games)</h2>
<div class="scroll-wrap">
<table>
<thead><tr><th>Player</th><th>Games</th><th>Flagged</th><th>Flag Rate</th></tr></thead>
<tbody>{player_rows}</tbody>
</table>
</div>

<h2>Top 200 Outlier Rows (sorted by PTS z-score)</h2>
<div class="scroll-wrap">
<table>
<thead>
<tr>
  <th>Player</th><th>Date</th><th>MP</th>
  <th>PTS</th><th>TRB</th><th>AST</th><th>STL</th><th>BLK</th><th>TOV</th>
  <th>Low MP</th><th>High MP</th><th>0-PTS</th><th>Stat OL</th>
  <th>z-PTS</th><th>z-TRB</th><th>z-AST</th><th>z-STL</th><th>z-BLK</th><th>z-TOV</th>
</tr>
</thead>
<tbody>{detail_rows}</tbody>
</table>
</div>

</body>
</html>"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding='utf-8')
print(f"Saved -> {OUT}")
