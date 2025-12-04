from collections import defaultdict
import math

class BigramLM:
    def __init__(self):
        self.unigram_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        self.vocab = set()

    def train(self, corpus):
        """
        corpus: list of tokens (e.g. ["I", "love", "NLP"])
        """
        for i in range(len(corpus)):
            w = corpus[i]
            self.unigram_count[w] += 1
            self.vocab.add(w)

            if i > 0:
                prev = corpus[i-1]
                self.bigram_count[(prev, w)] += 1

    def prob(self, prev, w):
        """
        Maximum Likelihood Estimate (no smoothing):
        P(w | prev) = count(prev, w) / count(prev)
        """
        if self.unigram_count[prev] == 0:
            return 0.0
        return self.bigram_count[(prev, w)] / self.unigram_count[prev]

    def predict_next(self, prev):
        """
        返回 prev 后最可能的下一个词
        """
        candidates = [(w, self.prob(prev, w)) for w in self.vocab]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]  # (word, prob)

# === 测试语料 ===
corpus = "i love natural language processing i love deep learning".split()

# === 训练模型 ===
lm = BigramLM()
lm.train(corpus)

# === 查看 bigram 概率 ===
print("P('love' | 'i') =", lm.prob("i", "love"))
print("P('deep' | 'love') =", lm.prob("love", "deep"))
print("P('language' | 'natural') =", lm.prob("natural", "language"))
print("P('i' | 'processing') =", lm.prob("processing", "i"))

# === 预测下一个词 ===
print("\nPrediction after 'i':", lm.predict_next("i"))
print("Prediction after 'love':", lm.predict_next("love"))
