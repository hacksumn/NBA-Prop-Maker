@echo off
cd /d "%~dp0"
python -m streamlit run vegas_gui.py --server.headless false --browser.gatherUsageStats false
