# System Memory Instructions

## Memory File Location
- Daily memories are stored in the `memory/` subdirectory
- Each day has one memory file named as `YYYY-MM-DD.md`

## How to Save Memories
- Use `save_today_memory` tool to save important information learned today
- Use `save_global_memory` tool to save permanent information (identity, user preferences)

## Memory Loading
- On startup, only today's and yesterday's memory files are loaded
- All memory files are prepended to the system prompt
