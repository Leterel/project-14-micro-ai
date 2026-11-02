"""Reproducible held-out comparison; prints JSON, never overwrites checkpoints."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from micro_ai import Model, pairs
from context_model import ContextModel
from dataset import read_dataset


def evaluate(steps=250, seed=42, corpus_path=None):
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError("steps must be 1–2000")
    training, validation, alphabet, digest = read_dataset(corpus_path or Path(__file__).with_name("corpus.txt"))
    counts = Counter(training)
    frequencies = {c: (counts[c]+1)/(len(training)+len(alphabet)) for c in alphabet}
    bigram, context = Model(alphabet, seed), ContextModel(alphabet, seed=seed)
    train_pairs = pairs(training, bigram)
    examples = context.examples(training)
    rng = random.Random(seed)
    for _ in range(steps):
        bigram.step(train_pairs, 8.0)
        batch = rng.sample(examples, min(64, len(examples)))
        context.step(batch, 1.0)
    # All losses score exactly the same targets, excluding the first three context characters.
    targets = validation[context.context:]
    held_pairs = pairs(validation, bigram)[context.context-1:]
    results = {
        "unigram": -sum(math.log(frequencies[c]) for c in targets)/len(targets),
        "bigram": bigram.loss(held_pairs),
        "context_tanh": context.loss(context.examples(validation)),
    }
    return {"seed": seed, "steps": steps, "corpus_sha256": digest,
            "training_characters": len(training), "held_out_targets": len(targets),
            "context": context.context, "hidden": context.hidden, "held_out_cross_entropy": results,
            "beats_unigram": {k: v < results["unigram"] for k, v in results.items() if k != "unigram"},
            "limitations": "Tiny original corpus, one paragraph held out; no general language-quality claim.",
            "sample": context.generate(training[:3], length=120, seed=seed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.steps, args.seed, args.corpus), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
