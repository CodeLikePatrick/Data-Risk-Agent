# Design

## Goal

Build a small but honest e-commerce risk system: a deterministic detection
engine plus an LLM review agent, with evaluation on synthetic labels.

## Layering

The detection engine and the agent are intentionally decoupled. The engine
produces structured signals and evidence; the agent consumes those signals and
produces a reviewable verdict. This mirrors production risk systems, where
detection, policy, and actioning are separate stages.

## Why two-stage detection

Rules give high precision, explainability, and fast emergency response. A model
generalizes beyond hand-written rules. Running rules first keeps the model
cheap and audit-friendly, and gives every positive case a rule or feature to
point to.

## Why a hand-written logistic regression

The model is intentionally small and dependency-free. This makes the project
reproducible and makes every implementation detail reviewable. For a demo with
six features, a linear model is enough; swapping in a tree model is an explicit
next step, not a hidden shortcut.

## Why ReAct with a tool registry

The tool registry keeps the model from touching data directly and makes the
system testable: tools can be replaced with mocks in unit tests. ReAct is used
because risk review is naturally multi-step (read signals, look up rules,
inspect a sample, decide).

## Evaluation stance

All reported numbers come from fixed-seed synthetic data with injected labels.
The benchmark is therefore reproducible and auditable, at the cost of not
reflecting real-world distribution. This tradeoff is documented rather than
hidden.
