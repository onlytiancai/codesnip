from gector.gec_model import GecBERTModel
from utils.helpers import get_target_sent_by_edits

# 初始化模型参数
model = GecBERTModel(
    vocab_path='data/output_vocabulary',            # 你的vocab路径
    model_paths=['models/roberta/roberta_1_gectorv2.th'],     # 你的模型路径
    weigths=[1.0],                                  # 单个模型的权重
    max_len=50,
    lowercase_tokens=True,
    iterations=3,
    min_error_probability=0.0,
    log=False
)

def process(input_sentence):
    print('Origin sentence:', input_sentence)
    # Step 1: 分词
    tokens = input_sentence.strip().split()

    # Step 2: 构造batch（注意这里需要列表嵌套）
    batch = [tokens]

    # Step 3: 推理，获取修改后的句子
    corrected_sentences, _ = model.handle_batch(batch)
    corrected_sentence = corrected_sentences[0]
    print("Corrected:", " ".join(corrected_sentence))

    # Step 4: 获取动作标签
    # 重跑一次推理以拿到 raw logits
    sequences = model.preprocess(batch)
    probs, idxs, error_probs = model.predict(sequences)

    vocab = model.vocab
    noop_index = vocab.get_token_index("$KEEP", "labels")

    token_actions = []
    for i, token in enumerate(["$START"] + tokens):  # GECToR 添加了 $START token
        label_idx = idxs[0][i]
        prob = probs[0][i]
        label = vocab.get_token_from_index(label_idx, "labels")

        # 跳过无改动的 token
        if label_idx == noop_index:
            continue

        action = model.get_token_action(token, i, prob, label)
        if action:
            start, end, act_token, act_prob = action
            token_actions.append({
                "token": token,
                "start": start,
                "end": end,
                "action": label,
                "confidence": act_prob
            })

    # 输出动作标签
    print("Actions:")
    for action in token_actions:
        print(action)
    print('======')


sentence_list = [
"She go to school every day",
'The list of item are on the table since yesterday.',
'She suggested me to go to the doctor immediatly.',
'He is married with a woman who lives in the same building and have two child.',
'I have experience of managing project and I enjoy to working with a team.',
'Although he don’t knew the answer, but he still tried to explaining it clearly.',
]

for sentence in sentence_list:
    process(sentence)
