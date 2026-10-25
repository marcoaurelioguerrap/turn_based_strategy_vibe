# NCM exact / legal-first-try experiment

Date: 2026-10-24
Persona: `aggressive`
Goal: raise the share of orders that are legal and parseable on the first try
(the league/dashboard `exact%` / `gen` share).

## What changed

- `train_generator.py`
  - New `--select {auto,legal,exact,loss}` checkpoint metric.
  - `auto` uses `legal_rate` when validation rows contain legal sets.
  - Legal rate now counts unparseable outputs as not legal.
  - Checkpoints record the selection metric; `--resume` resets `best` if the
    metric changes.
- `evaluate_orders.py`
  - Legal rate counts unparseable outputs as not legal.
  - New `--generator` flag for evaluating a specific checkpoint file.
- `persona_data.py`
  - Split summaries now report `legal_rows` and `legal_orders`.
- `build_finetune.py`
  - Reproducible mixed dataset: samples original rows (rare-action balanced)
    and weights DAgger rows by provenance (`generated` / `repaired` /
    `fallback`).
- Legal-annotated splits: `tbs_ncm/data/personas_legal_all/`.
- Stratified 500-row evaluation file:
  `artifacts/tbs/finetune_test/aggressive_stratified_test.jsonl`.

## Experiment

Evaluation set: 500 rows sampled evenly from the aggressive test split, across
all opponent personas. All rows carry the view's legal order set.

Fine-tunes:

1. **Small mix** — 1,500 original + 648 weighted DAgger rows, 3 epochs, lr 5e-5,
   best validation legal rate 14.84%.
2. **Fullmix** — 8,000 original + 648 weighted DAgger rows, 2 epochs, lr 2e-5,
   best validation legal rate 11.33%.

Both initialized from the release `generator.pt`; neither replaced the live
checkpoint.

## Results

| checkpoint | exact | parseable | legal first try |
|---|---:|---:|---:|
| baseline `generator.pt` | 2.00% | 97.60% | **8.40%** |
| previous `dagger_generator.pt` | 1.60% | 98.20% | 6.60% |
| new small legal-selected mix | 2.00% | 96.80% | 7.60% |
| new 8k original + DAgger fullmix | 1.20% | 95.40% | 7.60% |

Self-play 724-row check (aggressive vs aggressive) showed the same pattern:
baseline legal 10.64%, previous DAgger 6.77%, small legal-selected mix 5.66%.

## Conclusion

- Legal-rate checkpoint selection works mechanically (the correct epoch was
  saved), but the fine-tunes overfit the small mixed distribution: validation
  legal rate rose while held-out legal rate fell.
- **Do not promote the experimental checkpoints.** The release checkpoint is
  still the best on the representative test.
- More epochs on the same ~2k-row mix is not the answer. The next run needs
  substantially more on-policy states across all opponents, and the A/B must be
  positive before promotion.
- If data scaling still regresses, move to legal-prefix constrained decoding
  (mask generation against `legal_lines(view)`) instead of adding epochs.

## Artifacts

- `artifacts/tbs/dagger_generator_legal.pt` (experimental, not promoted)
- `artifacts/tbs/dagger_generator_fullmix.pt` (experimental, not promoted)
- `artifacts/tbs/personas/aggressive/generator.pt` (unchanged release checkpoint)
- `tbs_ncm/data/personas_legal_all/aggressive/{validation,test}.jsonl`
- `artifacts/tbs/finetune_test/aggressive_stratified_test.jsonl`
