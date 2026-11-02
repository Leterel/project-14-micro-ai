"""Train and sample the educational context network locally, without packages."""
import argparse
import json
from pathlib import Path
import random
from context_model import ContextModel
from dataset import read_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--corpus", type=Path, default=Path(__file__).with_name("corpus.txt"))
    sample = commands.add_parser("generate")
    sample.add_argument("--model", type=Path, required=True)
    sample.add_argument("--prompt", default="die ")
    sample.add_argument("--length", type=int, default=120)
    sample.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if not 1 <= args.steps <= 2000:
                parser.error("steps must be 1–2000")
            if args.output.exists():
                parser.error("output exists; choose a new filename")
            training, held_out, alphabet, _ = read_dataset(args.corpus)
            model = ContextModel(alphabet)
            examples = model.examples(training)
            validation = model.examples(held_out)
            before = model.loss(examples)
            rng = random.Random(42)
            for _ in range(args.steps):
                model.step(rng.sample(examples, min(64, len(examples))), 1.0)
            model.save(args.output)
            print(json.dumps({"steps": args.steps, "seed": 42, "train_before": before,
                              "train_after": model.loss(examples), "held_out": model.loss(validation),
                              "output": str(args.output), "architecture": "context-tanh"}))
        else:
            print(ContextModel.load(args.model).generate(args.prompt, args.length, args.seed))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
