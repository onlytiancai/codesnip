from paddlespeech.cli.tts import TTSExecutor

tts = TTSExecutor()

tts(
    text="It's the time of the year again, so I'd be interested hear what new (and old) ideas have come up. 又到了一年一度的这个时候，我很想知道大家又提出了哪些新想法（以及一些旧想法）。",               # 中英混合
    output="out.wav",              # 输出路径
    am="fastspeech2_mix",          # 官方中英混合声学模型
    voc="hifigan_csmsc",           # 通用 vocoder
    lang="mix",                    # 语言设置为 mix
)
