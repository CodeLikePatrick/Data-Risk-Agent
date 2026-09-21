# Decisions

This file records the main design choices and why they were made.

## Rules first, then a model

Rules give high precision, explainability and fast emergency response. A model
generalises beyond hand-written rules. Running rules first keeps the model cheap
and audit-friendly, and gives every flagged case a rule or feature to point to.

## A logistic regression written from scratch

The baseline model is small and dependency-free on purpose. This makes the
project reproducible and every implementation detail reviewable: the gradient
descent and normalisation are visible rather than hidden behind a library.

## A deterministic policy for the agent

The agent can run against a real LLM, but the benchmark uses a deterministic
policy that walks the identical tool sequence. That is what makes the reported
numbers reproducible and lets the LLM path be compared against a stable baseline.

## Synthetic data with injected labels

Synthetic data is used so every record has a known label. This trades realism for
exactness: precision, recall and false-positive rate can be measured directly.
The trade-off is documented rather than hidden.

## The threshold is a business decision

The score threshold is not tuned to a round number. It is chosen by the cost of a
false positive versus a false negative, and by the review capacity the business
has. The human-in-the-loop gate makes that capacity explicit.

## Human-in-the-loop for the tail

Clear predictions are automated; high and borderline scores are reviewed by a
person. Reviewing costs time, but it prevents the two mistakes that matter most:
blocking an innocent merchant and passing a risky one.
