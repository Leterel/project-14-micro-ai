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
Next: compare against a frequency baseline, enlarge the original corpus, then
implement a context window and hidden layer with gradient checks.
