#!/bin/bash

# Setup automated weekly retraining for NBA Prop Maker
# This script adds a cron job to retrain models every Sunday at 6 AM

echo "Setting up automated retraining..."

# Get the absolute path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Project directory: $PROJECT_DIR"

# Create cron job
CRON_JOB="0 6 * * 0 cd $PROJECT_DIR && /usr/bin/python3.11 scripts/utils/auto_retrain.py >> logs/retrain_\$(date +\%Y\%m\%d).log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "auto_retrain.py"; then
    echo "Cron job already exists. Updating..."
    # Remove old job
    crontab -l 2>/dev/null | grep -v "auto_retrain.py" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✓ Cron job added successfully!"
echo ""
echo "Schedule: Every Sunday at 6:00 AM"
echo "Command: python scripts/utils/auto_retrain.py"
echo "Logs: logs/retrain_YYYYMMDD.log"
echo ""
echo "To view cron jobs: crontab -l"
echo "To remove cron job: crontab -e (then delete the line)"
echo ""
echo "Manual retraining: python scripts/utils/auto_retrain.py"
