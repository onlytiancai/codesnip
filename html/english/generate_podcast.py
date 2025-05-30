import json
import os
import argparse
from dotenv import load_dotenv
from google import genai
from pathlib import Path
import contextlib
import wave
from google.genai import types


@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

def save_audio_blob(fname, response):
  blob = response.candidates[0].content.parts[0].inline_data
  with wave_file(fname) as wav:
    wav.writeframes(blob.data)



def generate_podcast_from_dialogue(json_file_path, output_audio_path=None):
    """
    Generate a podcast script and audio from a dialogue JSON file.
    
    Args:
        json_file_path (str): Path to the dialogue JSON file
        output_audio_path (str, optional): Path to save the audio file
    
    Returns:
        str: Path to the generated audio file
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    
    client = genai.Client(api_key=api_key)
    
    # Load the dialogue JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)
    
    # Extract dialogue number for naming
    dialogue_number = dialogue_data.get('number', '0')
    
    # If output path not specified, create one based on dialogue number
    if not output_audio_path:
        output_dir = Path("generated_podcasts")
        output_dir.mkdir(exist_ok=True)


    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker='Speaker 1',
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name='Zephyr',
                            )
                        )
                    ),
                    types.SpeakerVoiceConfig(
                        speaker='Speaker 2',
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name='Puck',
                            )
                        )
                    ),
                ]
            )
        )
    )

    # Process each scenario and dialogue
    for i, scenario in enumerate(dialogue_data.get('scenarios', [])):
        scenario_title = scenario.get('title', {}).get('en', 'Untitled Scenario')
        print(f"Processing scenario: {i} {scenario_title}")
        
        for j,dialogue in enumerate(scenario.get('dialogues', [])):
            output_audio_path = output_dir / f"dialogue_{dialogue_number}_{i}_{j}_podcast.wav"
            dialogue_title = dialogue.get('title', 'Untitled Dialogue')
            print(f"Processing dialogue: {j} {dialogue_title}")
            
            # Format the script for TTS
            script_lines = []
            
            # Process each exchange in the dialogue
            speakerMap = {}
            sparkerIndex = 1
            for exchange in dialogue.get('exchanges', []):
                speaker = exchange.get('speaker', 'Speaker')
                if speaker not in speakerMap:
                    speakerMap[speaker] = sparkerIndex
                    sparkerIndex += 1
                spakerNo = speakerMap[speaker]
                text = exchange.get('text', '')
                
                # Format as "Speaker X: Text"
                script_lines.append(f"Speaker {spakerNo}: {text}")
            
            # Join all exchanges with newlines
            script = "\n\n".join(script_lines)
            print(script)
            
            MODEL_ID = "gemini-2.5-flash-preview-tts" # @param ["gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts"] {"allow-input":true, isTemplate: true}
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents="TTS the following conversation between Speaker 1 and Speaker 2:\n\n "+ script,
                config=config,
            )

            save_audio_blob(str(output_audio_path), response)
            
            print(f"Generated podcast saved to: {output_audio_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate podcast from dialogue JSON")
    parser.add_argument("json_file", help="Path to the dialogue JSON file")
    parser.add_argument("--output", "-o", help="Path to save the output audio file")
    
    args = parser.parse_args()
    
    generate_podcast_from_dialogue(args.json_file, args.output)