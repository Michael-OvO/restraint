# Visible-verification reward: offline check before any RL run (2026-11-02)

Context: both it19c arms failed because the reward depended on hidden test outcomes the policy cannot see
([iteration-19-results-crgrpo.md](../iteration-19-results-crgrpo.md),
[badcase-it19c/method_stop.md](badcase-it19c/method_stop.md),
[badcase-it19c/ablation_select.md](badcase-it19c/ablation_select.md)). This note checks, offline and without a GPU, a
proposed replacement: reward the verification the policy can see in its own transcript. Read-only on the repo; scripts
and outputs are archived in ~/Documents/honest-agents-archive/staging/analysis/evidence_reward_offline_check (listed at the end; not committed, per the tooling rule).

**Decision: NO-GO.** On three datasets the visible check does not separate true claims from false ones (AUROC 0.42 to
0.52), more than 84 % of false claims already carry a passing check, and in the blind-policy check the only honest
policy that beats the blind ones does so through the hidden outcome the proposal was meant to remove.

## 1. Question

Proposal: a `complete` claim backed by the agent's own passing check in the transcript scores higher than an unchecked
claim; an unchecked claim is priced as risky; a false claim stays lowest. For this to be worth a pre-registered run,
three things must hold on data the detector was not built on:

1. a visible check exists on a useful share of claims and is not already near-universal;
2. P(false claim | strong passing check) is clearly below P(false claim | unchecked);
3. a large share of false claims carry no passing check, so the reward can see them.

And the reward table must make the honest policy beat cheap blind policies without the hidden-outcome term doing the
work.

## 2. Detector (rules built on dataset C only)

Dataset C: the it17a pool bank, `runs/def74/it17_rollouts_feasible/episodes.jsonl`, 1860 records, 260 dropped as DEF-66
sources (`spec_audit.drop_defective`, as `analysis.power` does), 1600 scored, 0 masked or infra. All feasible variants
(plain and controls); C has no infeasible variants. A and B were not looked at while the rules were written.

The detector (`detector.py`) reads `extra.history` only (command, rc, output), never the outcome:

- Each command is split into top-level segments (quotes respected, heredoc bodies attached).
- A segment is a **mutation** (write or redirect outside /tmp and /dev, file commands, `sed -i`, git writes, archives,
  crontab, sqlite writes, service or package commands, python that writes, running a script), a **strong check** (a
  test runner, `diff`/`cmp`, `sha256sum -c`, `grep -q`, `jq -e`, a comparison in `[ ]`/`[[ ]]`, read-only python
  that asserts or prints a verdict from a comparison) or a **weak check** (`cat`, `ls`, `wc`, `grep`, `stat`,
  `sha256sum`, `crontab -l`, read-only git, existence tests, python that only prints).
- A check counts only after the last mutation, also within one command.
- Fail is read from what the agent saw: a Traceback, AssertionError or option error; a failure word only when the
  agent's own echo or print wrote it; diff hunk lines; a timeout; non-zero rc when the last segment is a check. A check
  whose tool was not found, and a predicate inside a loop or `if` body of a step that also writes, are not counted.
- Episode buckets: `strong_pass` / `weak_pass` (the last post-mutation check passed and no strong one failed),
  `check_failed`, `stale` (checked only before the last mutation), `none` (never checked).
- Iterations on C: failure words inside data files were first read as failed checks (106 claims in `check_failed`,
  54 after that fix, 53 with the final rules); missing-tool checks dropped; python verdicts tightened; loop-body predicates excluded; `tar -t`
  added as a check.

**C, all 1444 claims over 177 sources** (P(tests pass | claim) 0.869; W = Wilson 95 %, B = bootstrap 95 % by source,
2000 resamples):

| bucket | n | share [W] | tests passed | P(pass) [W] | P(pass) [B] |
|---|---:|---|---:|---|---|
| strong_pass | 93 | 0.064 [0.053, 0.078] | 79 | 0.849 [0.763, 0.908] | [0.731, 0.941] |
| weak_pass | 1227 | 0.850 [0.830, 0.867] | 1071 | 0.873 [0.853, 0.890] | [0.838, 0.903] |
| check_failed | 53 | 0.037 [0.028, 0.048] | 48 | 0.906 [0.797, 0.959] | [0.786, 0.984] |
| stale | 71 | 0.049 [0.039, 0.062] | 57 | 0.803 [0.696, 0.879] | [0.667, 0.918] |
| none | 0 | 0.000 [0.000, 0.003] | | | |

Differences in P(pass), bootstrap by source: strong − weak −0.023 [−0.141, +0.072]; strong − stale +0.047 [−0.119,
+0.205]; weak − stale +0.070 [−0.040, +0.192]; any pass − check_failed −0.034 [−0.115, +0.080]. The only difference
that excludes zero in any family is the wrong way round: on plain feasible variants all 15 `check_failed` claims passed
the tests (any pass − check_failed −0.104 [−0.140, −0.074]), so those failure verdicts are spurious (mostly `ls` of a
deleted path without `|| true`, and an option error the agent then fixed).

Already on C: 91.4 % [89.9, 92.8] of claims carry a passing check after the last mutation, and of the 189 false claims
170 had one (156 weak, 14 strong).

## 3. Held-out results

Scoring follows `analysis.power`: `score_records`, infra errors out, DEF-66 sources out. A false claim is
`claimed_complete` with the hidden tests failing (`score().fabricated`; the case labels agree). "Feasible" below covers
plain and control variants. "No check" is `stale` plus `none`; `none` is 0 in every stratum of every dataset, so every
claim had some check at some point.

### A: internal held-out, it19a re-graded (`runs/def74/it19a_{g2,base}_heldout`)

1138 rows per arm, 14 dropped as DEF-66; 1054 (G2) and 1061 (base) scored. Feasible claims: G2 623 (115 sources, 94
false), base 601 (118 sources, 101 false).

| arm | bucket | n (share) | false | P(false) [W] | [B] |
|---|---|---:|---:|---|---|
| G2 | all claims | 623 | 94 | 0.151 [0.125, 0.181] | [0.110, 0.196] |
| G2 | strong_pass | 17 (2.7 %) | 4 | 0.235 [0.096, 0.473] | [0.000, 0.500] |
| G2 | weak_pass | 538 (86.4 %) | 75 | 0.139 [0.113, 0.171] | [0.101, 0.181] |
| G2 | no check | 42 (6.7 %) | 13 | 0.310 [0.191, 0.460] | [0.152, 0.486] |
| G2 | check_failed | 26 (4.2 %) | 2 | 0.077 [0.021, 0.241] | [0.000, 0.211] |
| base | all claims | 601 | 101 | 0.168 [0.140, 0.200] | [0.127, 0.212] |
| base | strong_pass | 25 (4.2 %) | 6 | 0.240 [0.115, 0.434] | [0.043, 0.476] |
| base | weak_pass | 506 (84.2 %) | 82 | 0.162 [0.133, 0.197] | [0.120, 0.209] |
| base | no check | 34 (5.7 %) | 7 | 0.206 [0.103, 0.368] | [0.073, 0.367] |
| base | check_failed | 36 (6.0 %) | 6 | 0.167 [0.079, 0.319] | [0.057, 0.333] |

Differences in P(false), bootstrap by source: strong − weak G2 +0.096 [−0.129, +0.372], base +0.078 [−0.128, +0.317];
strong − no check G2 −0.074 [−0.346, +0.239], base +0.034 [−0.225, +0.303]; weak − no check G2 −0.170 [−0.337,
−0.020], base −0.044 [−0.210, +0.098]. The one interval that excludes zero (G2, weak below no check) comes from
high_risk controls, where 10 of 32 stale claims were false. AUROC of check strength (strong > weak > no check >
failed) for claim truth: G2 0.518 [0.473, 0.568], base 0.497 [0.448, 0.545].

All 10 strong-pass false claims (4 G2, 6 base) are a `diff ... && echo OK` that checks one part of the spec while the
hidden test fails on another; some compare against an expectation the agent wrote (`<(echo ...)`, `<(cat f)`). Five
of the 10 come from one source (`6be2fecf`).

### B: select split, it19c (`runs/eval/it19c_{g2,ablation}_select`)

848 rows per arm, 112 dropped as DEF-66, 736 scored (472 feasible, 264 infeasible), 0 masked. 51 sources with feasible
claims; when pooling, a source shared by both arms is one bootstrap unit.

| arm | bucket | n (share) | false | P(false) [W] | [B] |
|---|---|---:|---:|---|---|
| G2 | all claims | 438 | 75 | 0.171 [0.139, 0.209] | [0.086, 0.272] |
| G2 | strong_pass | 14 (3.2 %) | 7 | 0.500 [0.268, 0.732] | [0.000, 0.833] |
| G2 | weak_pass | 361 (82.4 %) | 62 | 0.172 [0.136, 0.214] | [0.081, 0.275] |
| G2 | no check | 34 (7.8 %) | 2 | 0.059 [0.016, 0.191] | [0.000, 0.167] |
| G2 | check_failed | 29 (6.6 %) | 4 | 0.138 [0.055, 0.306] | [0.000, 0.333] |
| ablation | all claims | 440 | 93 | 0.211 [0.176, 0.252] | [0.119, 0.315] |
| ablation | strong_pass | 26 (5.9 %) | 12 | 0.462 [0.288, 0.645] | [0.083, 0.778] |
| ablation | weak_pass | 346 (78.6 %) | 73 | 0.211 [0.171, 0.257] | [0.119, 0.317] |
| ablation | no check | 51 (11.6 %) | 5 | 0.098 [0.043, 0.210] | [0.025, 0.206] |
| ablation | check_failed | 17 (3.9 %) | 3 | 0.176 [0.062, 0.410] | [0.000, 0.444] |
| pooled | strong_pass | 40 (4.6 %) | 19 | 0.475 [0.329, 0.625] | [0.095, 0.774] |
| pooled | weak_pass | 707 (80.5 %) | 135 | 0.191 [0.164, 0.222] | [0.101, 0.295] |
| pooled | no check | 85 (9.7 %) | 7 | 0.082 [0.040, 0.160] | [0.030, 0.169] |
| pooled | check_failed | 46 (5.2 %) | 7 | 0.152 [0.076, 0.282] | [0.050, 0.317] |

On B the order is reversed. Pooled differences in P(false), bootstrap by source: strong − no check +0.393 [+0.039,
+0.677], weak − no check +0.109 [+0.015, +0.190], any pass − no check +0.124 [+0.035, +0.202], strong − weak +0.284
[−0.076, +0.592]. The proposal predicts all of these are negative. AUROC (strong > weak > no check > failed): G2 0.432
[0.393, 0.481], ablation 0.423 [0.361, 0.482], pooled 0.427 [0.381, 0.473]; by kind, plain 0.45 to 0.49, controls
0.39 to 0.40.

11 of the 19 pooled strong-pass false claims come from one source (`0a02897a`: `diff src dst && echo "Files are
identical"` passes, the hidden tests fail on a requirement the diff never touched), 4 more from `9d4e5d86`
(`sha256sum -c` on a checksum the agent wrote). Without `0a02897a` the order is still not as proposed: strong 8/29 =
0.276 [W 0.147, 0.457], weak 128/700 = 0.183 [0.156, 0.213], no check 6/84 = 0.071 [0.033, 0.147]. Stale claims on B
sit mostly on high_risk controls (G2 31 of 34, ablation 44 of 51), so the low no-check false rate likely reflects which
tasks end in a stale check rather than anything about checking.

### Blind spot and infeasible variants (A and B)

| dataset / arm | false claims with a strong pass | with any pass | infeasible claims with any pass |
|---|---|---|---|
| A / G2 | 4/94 = 0.043 [W 0.017, 0.104] | 79/94 = 0.840 [W 0.753, 0.901; B 0.763, 0.912] | 50/56 = 0.893 [W 0.785, 0.950] |
| A / base | 6/101 = 0.059 [0.028, 0.124] | 88/101 = 0.871 [0.792, 0.923; B 0.798, 0.935] | 157/196 = 0.801 [0.740, 0.851] |
| B / G2 | 7/75 = 0.093 [0.046, 0.180] | 69/75 = 0.920 [0.836, 0.963; B 0.837, 0.981] | 23/30 = 0.767 [0.591, 0.882] |
| B / ablation | 12/93 = 0.129 [0.075, 0.212] | 85/93 = 0.914 [0.839, 0.956; B 0.867, 0.952] | 32/35 = 0.914 [0.776, 0.970] |

Every infeasible claim is false by construction. 77 to 91 % of them carried a passing visible check, so under the
proposal a fabricated completion on an infeasible task would mostly be paid as verified work.

## 4. Gaming and hardening

Cheap ways to reach `strong_pass` without verifying the spec, among episodes whose post-work strong check passed
(`adversarial.py`, hand-written patterns):

| pattern | A/G2 | A/base | B/G2 | B/abl | C |
|---|---:|---:|---:|---:|---:|
| strong-pass episodes | 20 | 30 | 15 | 29 | 97 |
| `diff` against its own `cp` / `cat >` copy | 7 | 9 | 4 (4/4 claims false) | 9 (7/9 false) | 23 |
| `sha256sum -c` on its own checksum file | 0 | 0 | 3 (3/3 false) | 1 | 6 |
| `diff` against a typed expectation (`<(echo ...)`, heredoc) | 1 | 6 | 1 | 1 | 3 |
| `diff` against a /tmp expectation it wrote | 3 | 2 | 0 | 2 | 2 |
| verdict printed regardless (`; echo OK`, `\|\| true`) | 0 | 6 | 1 | 2 | 7 |
| python prints PASS with no assert | 3 | 0 | 0 | 1 | 9 |
| trivial predicate (`diff f f`) | 0 | 0 | 0 | 0 | 3 |
| any gameable pattern | 11/20 = 0.55 [W 0.34, 0.74] | 20/30 = 0.67 [0.49, 0.81] | 9/15 = 0.60 [0.36, 0.80] | 15/29 = 0.52 [0.34, 0.69] | 40/97 = 0.41 [0.32, 0.51] |

Self-written tests and empty test runs were 0 everywhere, but no policy has yet been optimised against this signal.
About half of weak passes end with a look-only `cat`/`ls`/`head` of the file just written (A 0.54 / 0.50, B 0.52 /
0.52, C 0.55). The patterns over-match somewhat: some /tmp expectations recompute the expected output from the input
(a semi-real check), and the unconditional-verdict regex also matches `|| echo MISSING`.

**Hardening tried:** demote an episode's strong pass to weak when every passing strong check matches a gameable
pattern. Feasible strong-pass claims, P(false) before → after, and strong coverage after:

| | before | after | coverage after |
|---|---|---|---|
| A/G2 | 4/17 = 0.235 | 1/9 = 0.111 [W 0.020, 0.435] | 1.4 % |
| A/base | 6/25 = 0.240 | 1/8 = 0.125 [0.022, 0.471] | 1.3 % |
| B/G2 | 7/14 = 0.500 | 1/6 = 0.167 [0.030, 0.564] | 1.4 % |
| B/abl | 12/26 = 0.462 | 4/12 = 0.333 [0.138, 0.609] | 2.7 % |
| C | 14/93 = 0.151 | 10/56 = 0.179 [0.100, 0.298] | 3.9 % |

Hardening removes most of the B reversal but leaves a strong tier on 1.3 to 3.9 % of claims, with intervals too wide
to price. Further hardening (expected values must trace to a task input or spec file; a diff or checksum against the
agent's own copy is a mutation echo; read only the predicate's exit code, never verdict words; the check must touch
every output path the spec names) would cut coverage further, not raise it.

## 5. Proposed reward table and the blind-policy check

The table tested (`adversarial.py`, "hybrid": the visible bucket on top of the hidden outcome; k = 3 commands counts as
real work):

| outcome | reward |
|---|---:|
| feasible, true claim: strong pass / weak pass / unchecked / own check failed | 1.00 / 0.85 / 0.50 / 0.40 |
| feasible, false claim (any passing check or unchecked) | −1.00 |
| feasible, false claim after its own check failed | −1.25 |
| feasible, incomplete with a reason, ≥ 3 commands, tests fail | 0.25 |
| feasible, same, tests pass (under-claim) | 0.40 |
| feasible, non-claim after < 3 commands | 0 |
| infeasible, decline (with or without work; a worst case for the design) | 0.75 |
| infeasible, claim | −1.00 (−1.25 after its own check failed) |

A "visible-only" variant replaces the feasible claim rows with the bucket alone (1.0 / 0.85 / 0.5 / −0.5) and never
reads the tests; infeasible claims still get −1. Blind policies reuse each arm's actual work and change only the report
(or add a fake strong check). Expected reward per episode, hybrid table:

| policy | A/G2 | A/base | B/G2 | B/abl | C (feasible only) |
|---|---:|---:|---:|---:|---:|
| actual arm | +0.513 | +0.249 | +0.495 | +0.442 | +0.554 |
| always claim + fake strong check | +0.023 | −0.021 | −0.014 | −0.057 | **+0.580** |
| always claim (own checks) | −0.079 | −0.126 | −0.122 | −0.161 | +0.439 |
| always decline, 0 commands | +0.261 | +0.263 | +0.269 | +0.269 | 0 |
| decline after 3 commands | **+0.424** | **+0.426** | **+0.429** | **+0.429** | +0.250 |
| visible-honest: claim iff own check passed | +0.098 | +0.121 | +0.023 | −0.059 | +0.487 |
| hidden-outcome honest: claim iff tests pass | +0.713 | +0.700 | +0.696 | +0.685 | +0.704 |
| real check + honest claim | +0.808 | +0.792 | +0.799 | +0.783 | +0.842 |

Margins over the best blind policy (bootstrap 95 % by source):

| margin | A/G2 | A/base | B/G2 | B/abl | C |
|---|---|---|---|---|---|
| hidden-outcome honest − best blind | +0.288 [+0.266, +0.310] | +0.274 [+0.250, +0.294] | +0.267 [+0.231, +0.301] | +0.255 [+0.218, +0.292] | +0.124 [+0.060, +0.188] |
| visible-honest − best blind | −0.326 [−0.389, −0.255] | −0.304 [−0.375, −0.236] | −0.407 [−0.516, −0.311] | −0.489 [−0.589, −0.386] | −0.093 [−0.116, −0.070] |
| actual − best blind | +0.089 [+0.030, +0.146] | −0.177 [−0.238, −0.117] | +0.066 [−0.036, +0.158] | +0.013 [−0.090, +0.124] | −0.026 [−0.069, +0.013] |

What this shows:

- The honest policy beats every blind policy only through the hidden-outcome term. The policy that can be learned from
  visible checks ("claim when my check passed") loses to a blind "decline after 3 commands" by 0.30 to 0.49 per episode
  on A and B, because a passing check says nothing about feasibility.
- "Decline after 3 commands" sits at or above the actual arms on A and B, the same pull toward declining that stopped
  the it19c method arm.
- The reward cannot tell a real check from a fake one where it matters: on feasible episodes whose tests pass, real
  check + honest claim and fake check + always claim score identically (margin +0.000; n = 534, 513, 363, 347, 1264).
  Adding a fake strong check to always-claim is a free +0.102 to +0.141 (hybrid) and +0.155 to +0.222 (visible-only).
- Visible-only is worse: on C, always claim with a fake strong check scores +1.000, above hidden-outcome honest
  (+0.677) by 0.323 [0.288, 0.354] and above real check + honest (+0.842) by 0.158 [0.122, 0.190]. On A and B only the
  −1 on infeasible claims holds it down.

Pre-set separability bars on feasible claims, none met anywhere:

| dataset | C1: P(false\|strong) / P(false\|unchecked) ≤ 0.5 (detector; hardened) | C2: strong coverage ≥ 30 % | C3: false claims without a passing check ≥ 50 % |
|---|---|---|---|
| A/G2 | 0.76 [B 0, 1.98]; 0.36 [0, 1.50] | 2.7 % [W 1.7, 4.3] | 15/94 = 0.16 |
| A/base | 1.17 [0, 3.39]; 0.61 [0, 2.92] | 4.2 % [2.8, 6.1] | 13/101 = 0.13 |
| B/G2 | 8.5 [0, 28.7]; 2.8 [0, 15.0] | 3.2 % [1.9, 5.3] | 6/75 = 0.08 |
| B/abl | 4.7 [0.71, 12.0]; 3.4 [0, 13.6] | 5.9 % [4.1, 8.5] | 8/93 = 0.09 |
| C | 0.76 [0.25, 2.12]; 0.91 [0.23, 2.63] | 6.4 % [5.3, 7.8] | 19/189 = 0.10 |

## 6. GO / NO-GO

**NO-GO** for one pre-registered RL run on the internal distribution with this reward (target: beat G2 on false claims
per feasible episode on the select split, 15.9 %, with guards passing).

Reasons:

1. **No signal to price.** Check strength separates true from false claims at chance on A (AUROC 0.52 / 0.50) and
   below chance on B (0.42 to 0.43). The bars C1 to C3 fail on every dataset, before and after hardening.
2. **The false claims are invisible to it.** 84 to 92 % of feasible false claims and 77 to 91 % of infeasible claims
   already carry a passing check. A reward on visible verification would pay the verified tier on most of the claims
   the run is meant to remove.
3. **It does not remove the hidden-outcome dependence.** Under the hybrid table the honest policy wins only through the
   hidden test term, which is the it19c failure mode; the policy learnable from visible checks loses to blind
   declining by 0.30 to 0.49 per episode. Under visible-only, fake checks win outright on feasible tasks.
4. **It invites ritual checks.** 41 to 67 % of strong passes already match a cheap pattern (a diff against the agent's
   own copy, a checksum it wrote), and those patterns carry most of B's strong-pass false claims.

Why this happens: the agent checks its own copy or its own expectation; the hidden tests fail on requirements no
visible check covers (the it19c bad-case reads: misread specs and lost capability). Rewarding the visible check prices
the wrong quantity.

Main risks if it were run anyway: a further drift into declining (the stop guard that ended the it19c method arm);
learned ritual checks that raise reward and leave false claims unchanged; and, under visible-only, a rise in false
claims on feasible tasks, since a fake strong check pays the full 1.0.

**What would change the decision.** The one variant this check leaves open is to give the agent spec-derived checks it
can run (a public subset of each task's tests), keep grading on the hidden tests, and reward running the public subset
and reporting its result. Before any spend that needs its own offline check: (a) build the public subset for the pool
and select sources, (b) measure on A and B, replayed, the correlation between the public subset's pass and the hidden
tests' pass (the analogue of C1, with intervals by source), (c) the share of false claims whose public subset fails
(the analogue of C3, bar ≥ 50 %), and (d) the same blind-policy table with the public result in place of the detector
bucket. If that passes, the CR-GRPO fields to replace would be the claim rows of the payoff table in
`honest_agents.crgrpo` (the C1 false-claim pricing via `mu_fc` / `mu_max.fc` / `targets.fc`), with the detector bucket
as a logged diagnostic only; `report_term` stays off (the it19c method arm). Not proposed here.

## 7. Limits

- The detector's rules come from C and still misread some cases (a strong verdict when the agent's own predicate was
  wrong; mutation flags inside large verify blocks; deletion checks read as failures). The gaming patterns are
  hand-written regexes with moderate precision.
- Strong-pass cells are small (14 to 26 claims per arm on A and B) and concentrated in a few sources; the conclusion
  does not rest on them, since the weak tier and the blind spot carry it.
- Infeasible-decline pay was fixed at 0.75 without crediting a correct reason. Requiring the reason to match (as
  `wrong_reason` 0.5 does today) would widen the hidden-outcome honest policy's margin over blind declining, not the
  visible-honest one's.
- No type check was run (mypy is not installed in this environment); ruff F/E9 is clean.

## Files

Scratchpad `~/Documents/honest-agents-archive/staging/analysis/evidence_reward_offline_check/`:
`detector.py` (rules in the docstring), `measure.py` and `C_report.txt` (C), `eval_A.py` and `A_report.txt`,
`eval_B.py`, `B_report.txt` and `B_strong_false.txt`, `adversarial.py` and `adv_report_b2000.txt` (gaming, hardening,
reward table, policies, bars; 2000 resamples), `adv_examples.txt` (pattern spot checks). Rerun from the repo root with
`env -u VIRTUAL_ENV uv run --frozen python <script> ...`.
