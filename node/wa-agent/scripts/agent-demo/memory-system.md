# System Memory Instructions

## Memory Tools
- Use `read_memory` to view current memory content before writing
- Use `save_memory` to persist important information
- **Important**: Always call `read_memory` first to understand existing memory before calling `save_memory`
- **Important**: `save_memory` expects the FULL new content. You must compose and write the complete updated memory, not just new information to append
