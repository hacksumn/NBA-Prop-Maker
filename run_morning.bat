@echo off
cd /d "C:\Users\jakep\Downloads\Fresh_Start_NBA"
echo [%date% %time%] Starting morning NBA prop run... >> logs\morning_task.log

python run_daily.py >> logs\morning_task.log 2>&1
python files\nba_props.py predict >> logs\morning_task.log 2>&1

echo [%date% %time%] Morning run complete. >> logs\morning_task.log
