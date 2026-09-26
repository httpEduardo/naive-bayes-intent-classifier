import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class NaiveBayesIntent:
    def __init__(self):
        self.class_priors = {}
        self.word_counts = defaultdict(Counter)
        self.class_totals = Counter()
        self.vocab = set()

    def train(self, samples):
        class_counts = Counter(sample["intent"] for sample in samples)
        total = sum(class_counts.values())
        self.class_priors = {cls: count / total for cls, count in class_counts.items()}

        for sample in samples:
            intent = sample["intent"]
            tokens = tokenize(sample["text"])
            self.word_counts[intent].update(tokens)
            self.class_totals[intent] += len(tokens)
            self.vocab.update(tokens)

    def predict(self, text, top_k=3):
        tokens = tokenize(text)
        vocab_size = len(self.vocab) or 1
        scores = {}
        for intent, prior in self.class_priors.items():
            log_prob = math.log(prior)
            total_words = self.class_totals[intent]
            for token in tokens:
                count = self.word_counts[intent][token]
                prob = (count + 1) / (total_words + vocab_size)
                log_prob += math.log(prob)
            scores[intent] = log_prob
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
