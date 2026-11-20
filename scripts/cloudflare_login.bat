@echo off
REM One-time Cloudflare login for the daily dashboard upload (DEC-102). Double-click it from a normal Windows
REM window. Run it again only if the hosted dashboard stops updating and the log says the login is missing or expired.
REM
REM Do NOT run this from inside the Claude desktop app: that app is a packaged Windows app, and anything it writes
REM under AppData lands in a private folder the 07:30 scheduled task cannot see. This script stores the login in
REM %USERPROFILE%\.fresh_start_nba\cloudflare, outside AppData, and asks only for the three permissions a Pages
REM upload needs (account:read, user:read, pages:write).
REM The version below must match the one in scripts\project_operator.ps1 (a test checks this).
set "XDG_CONFIG_HOME=%USERPROFILE%\.fresh_start_nba\cloudflare"
if not exist "%XDG_CONFIG_HOME%" mkdir "%XDG_CONFIG_HOME%"
call npx --yes wrangler@4.164.0 login --scopes account:read user:read pages:write
echo.
echo If your browser showed "Successfully logged in", you are done and can close this window.
pause
