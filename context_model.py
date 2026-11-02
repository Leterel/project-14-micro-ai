"""Small, fully inspectable context network. No packages or pretrained weights."""
import math
import random
import json
from pathlib import Path
from micro_ai import softmax


class ContextModel:
    """One-hot context -> tanh hidden layer -> next-character softmax."""

    def __init__(self, alphabet, context=3, hidden=12, seed=42):
        if not isinstance(alphabet, str) or not 1 <= len(alphabet) <= 256 or len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet must contain 1–256 unique characters")
        if type(context) is not int or not 1 <= context <= 16 or type(hidden) is not int or not 1 <= hidden <= 128:
            raise ValueError("Invalid network dimensions")
        self.alphabet, self.context, self.hidden = alphabet, context, hidden
        self.index = {c: i for i, c in enumerate(alphabet)}
        rng = random.Random(seed)
        inputs = context * len(alphabet)
        self.parameters = {
            "input": [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(inputs)],
            "bias": [[0.0] * hidden],
            "output": [[rng.uniform(-0.2, 0.2) for _ in alphabet] for _ in range(hidden)],
            "output_bias": [[0.0] * len(alphabet)],
        }

    def examples(self, text):
        encoded = [self.index[c] for c in text]
        return [(encoded[i-self.context:i], encoded[i]) for i in range(self.context, len(encoded))]

    def forward(self, context):
        if len(context) != self.context or any(type(x) is not int or not 0 <= x < len(self.alphabet) for x in context):
            raise ValueError("Invalid context")
        p = self.parameters
        active = [i * len(self.alphabet) + x for i, x in enumerate(context)]
        hidden = [math.tanh(p["bias"][0][h] + sum(p["input"][i][h] for i in active)) for h in range(self.hidden)]
        logits = [p["output_bias"][0][j] + sum(hidden[h] * p["output"][h][j] for h in range(self.hidden)) for j in range(len(self.alphabet))]
        return active, hidden, logits

    def loss_and_gradients(self, examples):
        if not examples:
            raise ValueError("Need at least one example")
        p = self.parameters
        gradient = {name: [[0.0] * len(row) for row in matrix] for name, matrix in p.items()}
        loss = 0.0
        for context, target in examples:
            if type(target) is not int or not 0 <= target < len(self.alphabet):
                raise ValueError("Invalid target")
            active, hidden, logits = self.forward(context)
            top = max(logits)
            loss += top + math.log(sum(math.exp(x-top) for x in logits)) - logits[target]
            delta = [(x-(j == target))/len(examples) for j, x in enumerate(softmax(logits))]
            for j, value in enumerate(delta):
                gradient["output_bias"][0][j] += value
                for h in range(self.hidden):
                    gradient["output"][h][j] += hidden[h] * value
            for h in range(self.hidden):
                value = (1-hidden[h]**2) * sum(p["output"][h][j]*delta[j] for j in range(len(delta)))
                gradient["bias"][0][h] += value
                for i in active:
                    gradient["input"][i][h] += value
        return loss / len(examples), gradient

    def loss(self, examples):
        return self.loss_and_gradients(examples)[0]

    def step(self, examples, rate=0.5):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        loss, gradients = self.loss_and_gradients(examples)
        for name, matrix in self.parameters.items():
            for i, row in enumerate(matrix):
                for j in range(len(row)):
                    row[j] -= rate * gradients[name][i][j]
        return loss

    def generate(self, prompt, length=120, seed=42, temperature=0.8):
        if len(prompt) < self.context or type(length) is not int or not 0 <= length <= 10000:
            raise ValueError("Prompt must fill context; length must be 0–10000")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid temperature")
        encoded = [self.index[c] for c in prompt]
        rng = random.Random(seed)
        for _ in range(length):
            logits = self.forward(encoded[-self.context:])[2]
            encoded.append(rng.choices(range(len(self.alphabet)), weights=softmax([x/temperature for x in logits]))[0])
        return "".join(self.alphabet[i] for i in encoded)

    def save(self, path):
        data = {"version": 2, "architecture": "context-tanh", "alphabet": self.alphabet,
                "context": self.context, "hidden": self.hidden, "parameters": self.parameters}
        text = json.dumps(data, allow_nan=False)
        if len(text.encode("utf-8")) > 4_000_000:
            raise ValueError("Checkpoint exceeds 4 MB")
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(text)

    @classmethod
    def load(cls, path):
        path = Path(path)
        # Bound the read itself, including concurrent file growth.
        with path.open("rb") as handle:
            raw = handle.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("Checkpoint exceeds 4 MB")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("version") != 2 or data.get("architecture") != "context-tanh":
            raise ValueError("Invalid checkpoint architecture")
        model = cls(data.get("alphabet"), data.get("context"), data.get("hidden"))
        parameters = data.get("parameters")
        if not isinstance(parameters, dict) or set(parameters) != set(model.parameters):
            raise ValueError("Invalid parameter groups")
        for name, expected in model.parameters.items():
            matrix = parameters[name]
            if not isinstance(matrix, list) or len(matrix) != len(expected):
                raise ValueError("Invalid matrix dimensions")
            for row, wanted in zip(matrix, expected):
                if not isinstance(row, list) or len(row) != len(wanted):
                    raise ValueError("Invalid parameter row")
                if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000 for x in row):
                    raise ValueError("Invalid parameter value")
        model.parameters = parameters
        return model
