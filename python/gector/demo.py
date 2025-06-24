from gector.gec_model import GecBERTModel

# 初始化模型
model = GecBERTModel(
    vocab_path='data/output_vocabulary',
    model_paths=["models/roberta/roberta_1_gectorv2.th"],
    max_len=50,
    min_len=3,
    iterations=3,
    lowercase_tokens=True
)

# 测试句子（必须是token列表）
sentences = [
    "She no went to the market.".split(),
    "He go to school everyday.".split(),
    "This are incorrect sentence.".split()
]

# 执行纠错

corrected_sentences, num_corrections = model.handle_batch(sentences)

for i, (src, corrected) in enumerate(zip(sentences, corrected_sentences)):
    print(f"Original {i+1}: {' '.join(src)}")
    print(f"Corrected {i+1}: {' '.join(corrected)}")
    print("-" * 40)

print(f"Total corrections made: {num_corrections}")


def predict_with_actions_by_diff(model, sentence_tokens):
    corrected_batch, _ = model.handle_batch([sentence_tokens])
    corrected_tokens = corrected_batch[0]
    actions = []
    for src, tgt in zip(sentence_tokens, corrected_tokens):
        if src == tgt:
            actions.append('$KEEP')
        else:
            actions.append(f'change:{src}→{tgt}')
    return sentence_tokens, actions, corrected_tokens

sentence = "He go to school everyday .".split()
src, actions, corrected = predict_with_actions_by_diff(model, sentence)

for tok, act in zip(src, actions):
    print(f"{tok:10s} -> {act}")

print("Corrected:", " ".join(corrected))
