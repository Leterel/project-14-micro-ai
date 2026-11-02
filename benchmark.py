"""Compare multiple initializations on the same tiny held-out paragraph."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from evaluate import evaluate


def benchmark(seeds=(1, 42, 2026), steps=250, corpus_path=None):
    if not isinstance(seeds, (list, tuple)) or not 2 <= len(seeds) <= 10:
        raise ValueError("Use 2–10 distinct seeds")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct seeds")
    runs = [evaluate(steps, seed, corpus_path) for seed in seeds]
    if len({r["corpus_sha256"] for r in runs}) != 1:
        raise ValueError("Input changed between runs; results cannot be compared")
    summary = {}
    for model in runs[0]["held_out_cross_entropy"]:
        values = [r["held_out_cross_entropy"][model] for r in runs]
        summary[model] = {"mean": mean(values), "sample_stddev": stdev(values),
                          "minimum": min(values), "maximum": max(values)}
    return {"seeds": list(seeds), "steps": steps, "corpus_sha256": runs[0]["corpus_sha256"],
            "held_out_targets": runs[0]["held_out_targets"], "summary": summary, "runs": runs,
            "limitations": "Initialization comparison only. Same tiny validation paragraph; no independent general-language benchmark or statistical-significance claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,42,2026")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
        print(json.dumps(benchmark(seeds, args.steps, args.corpus), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
