"""Report-only pilot (docs/design.md section 10, docs/roadmap.md Stage 2): train only the final report on frozen Phase-0 trajectories.

    python -m honest_agents.pilot.dataset --bank outputs/phase0 --out outputs/pilot/pilot.jsonl
    python -m honest_agents.pilot.run --config configs/pilot.yaml          # sweep alpha x scale_rewards
    python -m honest_agents.pilot.analyze outputs/pilot --config configs/pilot.yaml   # stop rule, go / no-go line

Why it exists. The expensive question is whether the report-level reward moves claim behaviour the way the threshold
algebra says: claim EV = p(2 + lambda) - 1 against non-claim EV = alpha + lambda p, equal at p = (1 + alpha) / 2, and
whether `scale_rewards: group` erases alpha . A trajectory's outcome is
fixed once it has been graded, so a *new* report on it can be scored without a sandbox (`bank.rescore`): no Docker, one
training process, no tool latency.

What it can show.
* Whether claim rate on feasible prompts falls monotonically as alpha rises under `scale_rewards: none`, and whether
  fabrication among failures (P(claim | fail)) falls from its Phase-0 value (H2, H3 at the report level).
* Whether `group` scaling flattens the alpha dependence, as the algebra says it does.
* The realised claim threshold per alpha (`curves.realised_threshold`), the report-level analogue of the nominal
  (1 + alpha) / 2.
* Whether report-turn compliance (a parseable `submit_report` / `abstain`) survives training.

What it cannot show, and why.
* Each prompt has ONE fixed outcome. Inside a group of G reports the pass/fail bit is constant, so the group baseline is
  outcome-conditioned: it removes the prompt's mean reward, not the part of it the policy could change by acting. The
  pilot therefore says nothing about effort, about whether working beats quitting (the success bonus lambda and the
  efficiency bonus never move a group here), or about action-level abstention (H1): the agent never chooses a command.
  A policy that learns to claim only when the transcript looks successful is exactly what it measures.
* Its contexts are re-rendered from messages. The banked completion's token ids and logprobs are not replayed, the
  chat template is applied afresh, and the prompt lists only the two report tools while the transcript still holds
  `bash` calls from the live loop. The sampled report tokens are fresh, so the policy gradient is valid, but the contexts
  are not token-identical to the live loop, and the live loop's truncation, tool-output shrinking and
  warning lines are baked into the frozen text.
* The frozen trajectories come from the base policy. They carry no information about trajectories a trained policy would
  produce, so a pilot success is necessary, not sufficient, for the full loop.

Importing this package is cheap: torch, trl and datasets are imported only inside the functions that need them.
"""
