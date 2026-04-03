@echo off
cd /d "C:\Users\jakep\Downloads\Fresh_Start_NBA"
echo [%date% %time%] Starting morning NBA prop run... >> logs\morning_task.log

python run_daily.py >> logs\morning_task.log 2>&1

:: Run feature_pipeline + retrain advanced models every Sunday (day of week = 0)
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek.value__"') do set DOW=%%d
if "%DOW%"=="0" (
    echo [%date% %time%] Sunday -- rebuilding advanced features and retraining models... >> logs\morning_task.log
    python feature_pipeline.py >> logs\morning_task.log 2>&1
    python train_advanced_models.py >> logs\morning_task.log 2>&1
    python calibrate_confidence.py >> logs\morning_task.log 2>&1
    echo [%date% %time%] Advanced model rebuild complete. >> logs\morning_task.log
)

python nba_props.py predict >> logs\morning_task.log 2>&1

echo [%date% %time%] Morning run complete. >> logs\morning_task.log
