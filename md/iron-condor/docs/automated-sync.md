# Automated Wiki Synchronization Guide

Managing an LLM Wiki works best when it constantly reflects your background note-taking system. Instead of manually ingesting files every time you write something new, you can orchestrate an end-to-end automation pipeline.

This guide outlines a production-grade cron/launchd strategy for local Mac/Linux environments.

## The Two-Step Architecture

LLM Wiki Agent ingestion is a two-step process:
1. **Syncing to `raw/`**: Getting files from your personal vault/tools into the agent's staging area.
2. **Batch Ingestion**: Triggering `tools/ingest.py` on the synchronized directories to synthesize and weave them into the graph.

### Step 1: The Master Orchestrator Script

Create a comprehensive shell script in your wiki root (`daily-automated-sync.sh`):

```bash
#!/usr/bin/env bash
set -uo pipefail

# Define variables
LAB_DIR="$HOME/projects/active/personal-wiki-lab"
LOG_FILE="$LAB_DIR/automation-cron.log"
DATE=$(date "+%Y-%m-%d %H:%M:%S")

echo "=====================================================" >> "$LOG_FILE"
echo "[$DATE] Starting automated wiki synchronization..." >> "$LOG_FILE"

cd "$LAB_DIR" || exit 1

# 1. Run your personal Vault-to-Raw symlink script here
# Example: ./sync-raw.sh >> "$LOG_FILE" 2>&1

# 2. Trigger Litellm Batch Ingestion using LLM of your choice
export LLM_MODEL="gemini/gemini-3-flash-preview"
export GEMINI_API_KEY="AIzaSy..."  # or export OPENAI_API_KEY

echo "[$DATE] Batch ingesting markdown files..." >> "$LOG_FILE"
find raw/ -type l -name "*.md" -o -type f -name "*.md" | \
while read file; do 
    python3 tools/ingest.py "$file" >> "$LOG_FILE" 2>&1
done

# 3. Heal Graph Context (Auto-resolves broken semantic links)
echo "[$DATE] Healing broken nodes..." >> "$LOG_FILE"
python3 tools/heal.py >> "$LOG_FILE" 2>&1

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Automated sync completed." >> "$LOG_FILE"
echo "=====================================================" >> "$LOG_FILE"
```

Don't forget to make it executable: `chmod +x daily-automated-sync.sh`.

### Step 2: System Scheduler (macOS launchd)

For macOS, `launchd` is significantly more robust than `cron`.

Create a `.plist` file at `~/Library/LaunchAgents/com.personal-wiki-sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.personal-wiki-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/your-username/projects/active/personal-wiki-lab/daily-automated-sync.sh</string>
    </array>
    
    <!-- Execute automatically at 2:00 AM daily -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <!-- Run upon system boot if the interval was missed -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Diagnostic Logs -->
    <key>StandardOutPath</key>
    <string>/Users/your-username/projects/active/personal-wiki-lab/daemon.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/your-username/projects/active/personal-wiki-lab/daemon.stderr.log</string>
</dict>
</plist>
```

Load the daemon:
```bash
launchctl load ~/Library/LaunchAgents/com.personal-wiki-sync.plist
```

### Self-Healing & Health Monitoring
Since the automation runs silently at night, your `daemon.stderr.log` guarantees you will spot any API failures. The orchestrated script includes `tools/heal.py`, which is strongly recommended: it will seamlessly intercept and build concepts that accumulated throughout your day but were never individually formalized.
