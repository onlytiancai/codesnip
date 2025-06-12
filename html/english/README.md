# Podcast Generator

This script generates podcast audio from dialogue JSON files using Google's Gemini 2.5 Flash Preview TTS model.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

Generate a podcast from a dialogue JSON file:

```bash
python generate_podcast.py public/data/dialogue-1.json
```

Specify a custom output path:

```bash
python generate_podcast.py public/data/dialogue-1.json --output my_podcast.mp3
```

## Function Usage

You can also import and use the function in your own scripts:

```python
from generate_podcast import generate_podcast_from_dialogue

# Generate podcast from a dialogue file
audio_path = generate_podcast_from_dialogue("public/data/dialogue-1.json")
```

- https://ai.google.dev/gemini-api/docs?hl=zh-cn
- https://aistudio.google.com/prompts/new_chat
- https://aistudio.google.com/usage