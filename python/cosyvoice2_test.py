import sys
sys.path.append('third_party/Matcha-TTS')
import torchaudio
import torch
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from pydub import AudioSegment

cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
sample_rate = cosyvoice.sample_rate

prompt_speech_16k = load_wav('asset/zero_shot_prompt.wav', 16000)
prompt_speech_16k2 = load_wav('asset/case0013.wav', 16000)
cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'girl')
cosyvoice.add_zero_shot_spk('战场上的胜利，不仅依赖于兵力，更取决于决策的果敢和士气的高昂。', prompt_speech_16k2, 'boy')

# 扩展的双人对话，包含更多句子和情绪表达
dialogue = [
    {
        "text": "嗨，你今天看起来特别开心！[laughter]",
        "role": 'girl',
        "speaker_instruction": "用开心的语气说这句话"
    },
    {
        "text": "唉[breath]，是啊，但其实我有点难过。",
        "role": 'boy',
        "speaker_instruction": "用低落的语气说这句话"
    },
    {
        "text": "怎么了？发生什么事了？你可以告诉我。",
        "role": 'girl',
        "speaker_instruction": "用关切的语气说这句话"
    },
    {
        "text": "最近工作压力太大，感觉有点撑不住了。",
        "role": 'boy',
        "speaker_instruction": "用疲惫的语气说这句话"
    },
    {
        "text": "我懂，有时候真的需要好好休息一下，别太逼自己。",
        "role": 'girl',
        "speaker_instruction": "用温柔的语气说这句话"
    },
    {
        "text": "谢谢你，听你这么说，我心里舒服多了。[breath]",
        "role": 'boy',
        "speaker_instruction": "用感激的语气说这句话"
    },
    {
        "text": "要不要一起出去散散步？换个环境可能会好点。",
        "role": 'girl',
        "speaker_instruction": "用邀请的语气说这句话"
    },
    {
        "text": "好啊，听起来不错，我也需要放松一下。",
        "role": 'boy',
        "speaker_instruction": "用稍带笑意的语气说这句话"
    },    
]

tts_tensors = []

for turn in dialogue:
    text = turn["text"]
    instruction = turn["speaker_instruction"]
    role = turn['role']

    outputs = cosyvoice.inference_zero_shot(text, instruction, '', zero_shot_spk_id=role, stream=False)
    for j in outputs:
        tts_speech = j['tts_speech']
        tts_tensors.append(tts_speech.squeeze(0))

full_audio = torch.cat(tts_tensors, dim=0)

wav_path = "dialogue_full_long.wav"
torchaudio.save(wav_path, full_audio.unsqueeze(0), sample_rate)


# 转成 mp3
audio_segment = AudioSegment.from_wav(wav_path)
mp3_path = "dialogue_full_long.mp3"
audio_segment.export(mp3_path, format="mp3")

print(f"WAV saved to: {wav_path}")
print(f"MP3 saved to: {mp3_path}")
