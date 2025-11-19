# ADR 003 - No bi-encoder fine-tuning yet

## Status

Accepted (provisional; revisit when the eval set crosses ~500 labelled pairs).

## Context

The off-the-shelf `all-mpnet-base-v2` checkpoint was trained for general semantic similarity. Resume-to-JD matching is a more specialised distribution, and contrastive fine-tuning on in-domain triplets is the obvious lever.

## Decision

Do not fine-tune the bi-encoder in this version. Use the public pretrained checkpoint as-is.

## Reasoning

With the current eval set (5 resumes x 10 JDs = 50 graded pairs, of which only the 2+ grades are positives), there is not enough signal to fine-tune without overfitting:

- A typical sentence-transformers contrastive training run needs thousands of (anchor, positive, negative) triplets to move the encoder without catastrophic forgetting of general semantics.
- Fine-tuning on <100 triplets will memorise the small set and degrade out-of-distribution performance — which in this case means almost every real resume the project hasn't seen.
- The validation signal is too small to detect this regression; informal spot-checks would mistake memorisation for generalisation.

The higher-leverage move at this stage is to grow the labelled set, not to spend training budget on what already exists. The plan in priority order:

1. Migrate to a larger public corpus (~12k JDs, see data card).
2. Bootstrap a labelled pool via active learning: have the current bi+cross pipeline propose candidates, label the marginal pairs by hand.
3. Target ~500 (anchor, positive, hard-negative) triplets, then fine-tune `all-mpnet-base-v2` with MultipleNegativesRankingLoss for 1-2 epochs.
4. Hold out a frozen evaluation slice from the start; only unfreeze it for final reporting.

## Trigger to revisit

When the labelled set crosses ~500 triplets with at least one held-out validation slice, this ADR is replaced with an explicit fine-tuning plan and result.
