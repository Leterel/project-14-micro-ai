# Micro-AI: End-to-End Roadmap

Status: Planned. This roadmap expands the existing project without replacing the from-scratch educational model.
Updated: 2026-11-06

## Vision

Build a practical, ChatGPT-like local AI assistant as quickly as feasible, while separately continuing to learn how language models work by implementing and training our own small model. "Five minutes" means minimal friction and fast iteration, not a literal delivery promise or ChatGPT-level model training from scratch.

## Two tracks, one repository

**Track A: Micro-AI Core (from scratch)**
- Preserve the existing pure-Python character bigram model, training script, corpus and tests.
- Add a frequency baseline and more original training data.
- Add a context window, hidden layer and gradient checks.
- Explore tokenization, more capable architectures, training, evaluation and optimization.
- Document model limitations and reproducible experiments.

**Track B: Micro-AI Assistant (practical capabilities)**
- Integrate an existing capable model behind a replaceable provider interface, preferably local-first (for example Ollama); optional API providers require explicit configuration and may cost money.
- Build conversational UI/CLI with streaming, context management and persistent conversation history.
- Add document ingestion and RAG with source attribution.
- Add constrained tools (files, search, approved commands/APIs) with typed input/output contracts.
- Add multi-step agent orchestration with clear stop conditions and user confirmation for consequential actions.
- Add timeouts, bounded retries, fallback behavior, structured logs, tracing and regression tests.
- Add prompt-injection defenses, input validation, least-privilege permissions and secret handling.
- Add user-friendly error messages, uncertainty signals, settings, packaging and Windows/PowerShell documentation.

## Implementation milestones

- [ ] 0. Audit existing code, tests, dependencies and target Windows hardware; establish baseline.
- [ ] 1. Select local model based on actual RAM/VRAM and benchmark it; keep the existing educational model untouched.
- [ ] 2. Add a minimal working assistant chat loop using the selected model.
- [ ] 3. Add session context and persistent chat history.
- [ ] 4. Add file-based retrieval (RAG) with citations and retrieval quality tests.
- [ ] 5. Add tool interfaces with schemas, permissions and dry-run support.
- [ ] 6. Add safe multi-step task execution and approval gates.
- [ ] 7. Add reliability: timeout, retry/backoff, fallback and recovery.
- [ ] 8. Add security checks for prompt injection, data access and secret leakage.
- [ ] 9. Add observability and evaluation: trace IDs, logs, test tasks, success/latency/resource metrics.
- [ ] 10. Add a usable local UI and configuration.
- [ ] 11. Benchmark and optimize performance, resource usage and quality.
- [ ] 12. Package, document and test a reproducible local Windows setup.

## The seven agent engineering disciplines

1. System design: explicit components and data flow.
2. Tool and contract design: validated inputs/outputs and limited capabilities.
3. Retrieval engineering: relevant source selection, chunking and ranking.
4. Reliability engineering: controlled failures and recovery.
5. Security and safety: boundaries, approvals and attack resistance.
6. Evaluation and observability: measurable quality and traceable actions.
7. Product thinking: understandable, trustworthy user experience.

## Working rules

- Small, reversible changes; tests before/after each milestone.
- Never overwrite the original learning model or claim it is ChatGPT-equivalent.
- Do not install software, change local paths or select a model until actual environment constraints are checked.
- Prefer free, local-first tools when practical; distinguish software cost from hardware requirements.
- Keep Track A and Track B clearly separated in code and documentation.
- No arbitrary shell execution or irreversible actions without explicit approval.
- Mark each milestone done only after an actual successful test.

## Current verified state (README, 2026-11-06)

The educational model is a character bigram neural language model in pure Python with learned logits, softmax, cross-entropy and analytic gradients. It uses one character of context and is not a general chat model. The README states that chat, persistent memory, API, transformer and GPU training have not yet been implemented. The roadmap above is a plan, not completed functionality.
