# Research behind iteration 20, revision 4

These are the reports the revision-4 design ([../../iteration-20-rl-design.md](../../iteration-20-rl-design.md)) was
built from, on 2026-11-09 and 2026-11-10. They are records, kept as the agents returned them. Where two disagree,
the order of authority is: the design, then the six cross-examination reports, then the judges, then the rest.

Every new number is from train-side data (the select runs, the train-pool bank, archived RL training runs) or from
held-out aggregates already printed in earlier documents. No held-out episode was opened. Scripts are archived at
`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/` (`archive/staging/analysis/it20_rev4/` after `fetch`,
once the owner adds it to the train bundle).

## How they were produced

| step | reports | what each did |
|---|---|---|
| 1. Grill | `grill_method`, `grill_data`, `grill_metrics`, `grill_code`, `grill_history`, `grill_program` | six adversarial reviews of revision 3, one lens each |
| 1. Evidence | `evidence_rl-history`, `evidence_dynamics`, `evidence_measurement`, `evidence_literature` | every past RL run re-read and re-scored; the gradient's signal and noise; what the executor can measure; external results |
| 2. Research | `research_pass-fail-anatomy`, `research_failure-taxonomy`, `research_headroom-pool-model`, `research_twin-judgment-signal`, `research_throughput`, `research_algorithm-feasibility`, `research_lit-agentic-rl`, `research_lit-honesty-rl` | new analyses and transcript reads on train-side data; two literature surveys |
| 3. Cross-examination | `cross_method-adjudication`, `cross_measurement-adjudication`, `cross_dp-code`, `cross_dp-offline`, `cross_second-read`, `cross_literature` | each checked the earlier reports against each other, the code and the data, and settled the disagreements |
| 4. Designs | `designs/design_FREE`, `designs/design_SCALE`, `designs/design_DP` | three complete redesigns written independently from the verified evidence |
| 5. Judges | `judges` | three judges (effect, validity, feasibility) scored the designs, ranked them, and listed grafts and required fixes |

`lead_notes.md` is the lead's working synthesis between steps 2 and 3. Several of its claims were corrected by the
cross-examination (the corrections are listed in each cross report).

## What to read

1. `cross_method-adjudication.md` and `cross_measurement-adjudication.md`: the decisions on method and measurement.
2. `cross_second-read.md` and `research_failure-taxonomy.md`: what G2's false claims look like on train-side data.
3. `evidence_dynamics.md`: why the earlier RL run moved like a noise walk.
4. `cross_literature.md`: which external claims hold, which do not, and the prior art for decision-point training.
5. `judges.md`: why the consensus design won, and what the final plan had to fix.

## Cautions

- **The three designs in `designs/` are superseded.** They are candidates, not plans. `design_FREE.md` is the base of
  revision 4; the final design differs from it in every item the judges listed.
- The step-1 and step-2 reports were written before the cross-examination. Some of their numbers were corrected
  later, for example the share of false claims with at least three commands left (79 %, not 86 %) and two citations
  that must not be used (`cross_literature.md`, section 7).
- Reports from steps 1, 2 and 5 are structured dumps of what each agent returned (summary, facts or findings with
  sources, recommendations). The cross-examination reports and the designs are the agents' own documents.
- Probabilities in these reports are judgments stated as bets, not computations.
- Hygiene departures are recorded in the design's "Known contamination" paragraph.
