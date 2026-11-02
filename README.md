# PROJECT #15: MICRO AI FROM SCRATCH

Status: Concept / planning
Owner: Leterel
Created: 2026-10-29

## Vision

Build a small AI system from the ground up, understand its internals, train it and run it locally.

## Goals

- Understand neural networks and language models
- Implement a small neural network
- Build a tokenizer and prepare training data
- Train a small language model from scratch
- Implement context handling and conversation memory
- Build a local chat interface and API
- Run locally on Windows
- Optimize performance, memory usage and model quality
- Add tests, documentation and reproducible training steps
- Track all source code and progress in GitHub

## Principles

- Train our own small model instead of merely wrapping an existing LLM
- Frameworks such as PyTorch are allowed as development tools
- Start with a minimal working prototype and expand incrementally
- Document and test every milestone
- Keep the project local-first where practical

## Roadmap

1. Inspect hardware and define realistic model scope
2. Set up the development environment
3. Implement a minimal neural network
4. Build tokenization and a tiny training dataset
5. Train and evaluate the first model
6. Add text generation and a chat loop
7. Implement context and persistent memory
8. Build a usable local interface
9. Benchmark, test and document
10. Package the finished prototype

## Initial status

First educational milestone implemented on 2026-11-02: a character bigram
neural language model in pure Python, with learned logits, softmax,
cross-entropy and analytic gradients. It starts from random weights and trains
on the original text in `corpus.txt`. No downloaded model or paid service.

```powershell
python -m unittest -v
python micro_ai.py train --steps 1000 --output model.json
python micro_ai.py generate --model model.json --prompt "die "
```

Python 3.12 or newer; no package installation. Existing model files are never
overwritten. Checkpoints contain numbers and an alphabet, not executable pickle.
The final paragraph is held out from training. Reported losses are per next
character; this tiny corpus is not a representative language benchmark.

This model has only one character of context. Output is mostly invented
character sequences, not reliable German or answers. No chat, persistent memory,
API, transformer, GPU training or production model is implemented yet.
The original bigram interface remains available for comparison.

## Context network and reproducible evaluation

`context_model.py` adds a trainable three-character context window and a
12-unit tanh hidden layer. All calculations and backpropagation use Python's
standard library; no downloaded weights, API or GPU setup. The tests check
every parameter against finite differences and learn the context-dependent
`aabb` pattern that a one-character model cannot disambiguate.

```powershell
python -m unittest -v
python evaluate.py --steps 250 --seed 42
python context_cli.py train --steps 500 --output model-context.json
python context_cli.py generate --model model-context.json --prompt "die "
```

The evaluator trains on the first paragraphs only, learns its alphabet and
frequency baseline from training data, and compares all models on exactly the
same held-out targets. It prints corpus hash, seed, sample and losses as JSON;
it does not write or overwrite any checkpoint. The existing bigram checkpoint
format is unchanged. The separate context CLI uses validated numeric JSON
checkpoints (version 2, no pickle), bounded to 4 MB, and refuses overwrites.
A chat UI is not implemented. On the current tiny original corpus, seed 42 / 250 steps yielded
cross-entropies 2.714 (unigram), 1.974 (bigram), 1.455 (context network), lower
is better. This is one 72-target paragraph, not a general language benchmark.
Generated text remains unreliable character sequences, not factual answers.

## Entwicklung und Übergabe

[AGENTS.md](AGENTS.md) bündelt den Einstieg für Entwickler und KI-Agenten; [TESTING.md](TESTING.md) nennt reproduzierbare Prüfungen und deren Grenzen. Der aktuelle Übergabestand steht in [HANDOFF.md](HANDOFF.md).
