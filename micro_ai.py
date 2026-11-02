"""A tiny character bigram neural language model, using only Python's stdlib.

Each character selects a row of learned logits; softmax predicts the next
character. Cross entropy is optimized by analytic gradient descent. This is
an educational one-layer model, with one-character context, not an assistant.
"""
import argparse
import json
import math
from pathlib import Path
import random


def softmax(logits):
    top = max(logits)
    values = [math.exp(x - top) for x in logits]
    total = sum(values)
    return [x / total for x in values]


class Model:
    def __init__(self, alphabet, seed=42):
        if not isinstance(alphabet, str) or not 1 <= len(alphabet) <= 256 or len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet must contain unique characters")
        self.alphabet = alphabet
        self.index = {c: i for i, c in enumerate(alphabet)}
        rng = random.Random(seed)
        self.weights = [[rng.uniform(-0.01, 0.01) for _ in alphabet] for _ in alphabet]

    def encode(self, text):
        return [self.index[c] for c in text]

    def loss(self, pairs):
        if not pairs:
            raise ValueError("Need at least one pair")
        # Log-sum-exp avoids log(0) even for large learned logits.
        total = 0.0
        for x, y in pairs:
            row = self.weights[x]
            top = max(row)
            total += top + math.log(sum(math.exp(v - top) for v in row)) - row[y]
        return total / len(pairs)

    def step(self, pairs, rate):
        if not pairs or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Need pairs and a positive finite learning rate")
        gradient = [[0.0] * len(self.alphabet) for _ in self.alphabet]
        for x, y in pairs:
            probabilities = softmax(self.weights[x])
            for j, p in enumerate(probabilities):
                gradient[x][j] += (p - (j == y)) / len(pairs)
        for i, row in enumerate(gradient):
            for j, value in enumerate(row):
                self.weights[i][j] -= rate * value
        return gradient

    def generate(self, prompt, length=120, seed=42, temperature=0.8):
        if not isinstance(prompt, str) or not prompt or type(length) is not int or not 0 <= length <= 10000:
            raise ValueError("Need a prompt and length between 0 and 10000")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        self.encode(prompt)  # Reject unknown characters before generating.
        rng = random.Random(seed)
        text = prompt
        for _ in range(length):
            row = self.weights[self.index[text[-1]]]
            text += rng.choices(self.alphabet, weights=softmax([x / temperature for x in row]))[0]
        return text

    def save(self, path):
        # Exclusive creation protects existing checkpoints.
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump({"version": 1, "alphabet": self.alphabet, "weights": self.weights}, handle)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as handle:
            raw = handle.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("Checkpoint too large")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Invalid checkpoint object")
        alphabet = data.get("alphabet")
        if data.get("version") != 1 or not isinstance(alphabet, str) or not 1 <= len(alphabet) <= 256:
            raise ValueError("Invalid checkpoint alphabet or version")
        model = cls(alphabet)
        weights = data.get("weights")
        if not isinstance(weights, list) or len(weights) != len(alphabet):
            raise ValueError("Invalid matrix")
        for row in weights:
            if not isinstance(row, list) or len(row) != len(alphabet):
                raise ValueError("Invalid row")
            if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100 for x in row):
                raise ValueError("Invalid weight")
        model.weights = weights
        return model


def pairs(text, model):
    encoded = model.encode(text)
    return list(zip(encoded, encoded[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--output", type=Path, default=Path("model.json"))
    generate = commands.add_parser("generate")
    generate.add_argument("--model", type=Path, default=Path("model.json"))
    generate.add_argument("--prompt", default="die ")
    generate.add_argument("--length", type=int, default=120)
    generate.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "train":
        if not 1 <= args.steps <= 10000:
            parser.error("steps must be between 1 and 10000")
        if args.output.exists():
            parser.error("output already exists; choose a new filename")
        corpus = Path(__file__).with_name("corpus.txt").read_text(encoding="utf-8")
        paragraphs = corpus.strip().split("\n\n")
        training, validation = "\n".join(paragraphs[:-1]), paragraphs[-1]
        model = Model("".join(sorted(set(corpus))))
        data = pairs(training, model)
        before = model.loss(data)
        for _ in range(args.steps):
            model.step(data, 8.0)
        model.save(args.output)
        print(json.dumps({"parameters": len(model.alphabet) ** 2, "steps": args.steps,
                          "train_loss_before": before, "train_loss_after": model.loss(data),
                          "held_out_loss": model.loss(pairs(validation, model)),
                          "training_pairs": len(data), "held_out_pairs": len(validation)-1}))
    else:
        print(Model.load(args.model).generate(args.prompt, args.length, args.seed))


if __name__ == "__main__":
    main()
