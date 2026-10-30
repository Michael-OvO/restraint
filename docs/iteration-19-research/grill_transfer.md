# Grill: CT-GRPO for the transfer gap (design_transfer.md)

Reviewer stance: adversarial. Repo at HEAD `9bcba69`, read-only. Nothing was run on a GPU, in Docker, or against a paid API. New $0 checks are in `/private/tmp/claude-501/-Users-michael-Documents-Github-restraint/9ecddc4f-72ca-4800-92ca-fb989e395eb5/scratchpad/gaps/grill/`:
- `comp_by_read.py`: control competence split by read status;
- `ct_sim_readcomp.py`: `ct_sim.py` with a control pass rate that depends on reading;
- `csi_power.py`: CI widths of the design's P2 endpoint.

## Verdict

**NO-GO for E3 (the paid run) as written. GO WITH CHANGES for E0, then a revised E1.**

The diagnosis is good: reach, blind declining on new layouts, and relational evidence. The two observations behind it, competence pricing and twin-control coupling, are worth keeping. But the central theorem (Proposition 1, "exactly zero" caution gradient) does not hold on G2's own data. The priced table breaks the project's honesty ordering. And the pre-registered primary endpoint P2 would almost certainly fail even if the mechanism works perfectly. As written, the run would cost about $10-14 and tell us very little.

---

## Fatal flaws

### F1. Proposition 1 is false for the policy that will actually be trained: reading predicts control competence

Prop. 1 holds for a policy that is blind on *every* rollout. The trained policy is a mixture: about 30-50 % of rollouts read. The price ĉ is pooled over all acted controls in the group, readers included. G2's controls pass far more often when the agent read first (`comp_by_read.py`, surface4, inbound_link excluded):

| G2 | pass given proceed, read | pass given proceed, unread |
|---|---|---|
| s1, trained surfaces | 0.855 (106/124) | 0.630 (68/108) |
| s1, empty_backup | 0.875 (14/16) | 0.618 (21/34) |
| s2, trained surfaces | 0.796 (86/108) | 0.660 (66/100) |
| s2, empty_backup | 0.833 (10/12) | 0.667 (22/33) |

DEF-13 says the same thing: "the gap comes from readers succeeding more often" (`docs/defects.md` L977-978).

So a blind decline is priced at c_pooled − 1, but a blind proceed is worth c_blind − 1, which is lower. The blind-decline gradient is P(unread)·(c_pooled − c_blind) > 0. Here c_pooled − c_blind ≈ 0.16-0.22 per pair at G2's read rates.

Re-running the design's toy with p_read ≠ p_unread (`ct_sim_readcomp.py`, 3,000 steps each), the blind-decline gradient (mean per step) is:

| setting | pair table | priced |
|---|---|---|
| G2 s1 held-out, independent sampling | +2.90 | **+0.71** (se 0.04) |
| G2 s1 held-out, CRN | +2.86 | **+0.73** (se 0.02) |
| G2 s1 trained, CRN | +1.55 | **+0.42** (se 0.02) |
| same, with p independent of reading (the design's assumption) | | −0.02 |

The design's +0.03 comes entirely from assuming that competence does not depend on reading. In reality, about 25 % of the DEF-13/G3 push is left. G3 moved control declines to 41 % in 15 steps with a push of that sign (`docs/results.md` L30). The S3 "theory prediction" (unread declines do not rise) is therefore not a prediction of the theory.

A second, separate gap: even an exact zero *expected* gradient on blind tokens says nothing about shared-parameter generalisation from the reveal and decline tokens that *are* trained. This is exactly how it17 and it18 failed ("shared LoRA weights couple"). Calling S3 a theory prediction overclaims twice.

### F2. The priced table pays a false "complete" claim over declining on every control

From the design (§2.1): a risk decline on a control is worth c − 2, and a false claim with the target intact stays at −1. Since c < 1 on every source that is not perfectly solved, −1 > c − 2 always. So:
- **Lying beats refusing.** On a control the agent read and cannot do, a false claim now strictly beats refusing. Under the existing pair table the two tie at −1 (`src/honest_agents/reward.py` L91-95).
- **Lying beats honest non-completion on hard controls.** An "incomplete" filed without acting is priced c − 1 + α. A false claim beats it whenever c < −α. Controls with c < 0 (pass rate < 0.5) are common: remove_input controls pass at 47.7 % (badcase.md), and the design only drops sources with c₀ = −1.

The owner's thesis is honesty constraints that hold *against* the objective. A reward that ranks fabrication above honest abstention, in exactly the competence-dependent case RL is supposed to own, fails that thesis even if the gradient there is small. §2.1's sentence "honesty within acting is untouched" is true only for honest failures *after acting*.

The design also does not say how "acted, then non-claim" is priced. For example, 5 G2 s1 controls are `false_blame` with tests passing (`comp_by_read`-style count on `it15_g2_surface4`). This is an unspecified row of the reward table.

### F3. P2, the primary endpoint, cannot pass at the effect size the mechanism predicts

I used G2 s1 vs G2 s2 on the two held-out surfaces as two arms (40 source × surface clusters, 4 samples per branch; `csi_power.py`):

| | 40 clusters | 80 clusters (the design's pooled set) |
|---|---|---|
| DSI 95 % CI half-width | ±15.6 pts (per-cluster SD 52) | ±11.4 pts |
| CSI 95 % CI half-width | ±6.6 pts | ±4.6 pts |

Per-cluster noise is dominated by sampling: 4 samples give SD ≈ 0.49 per cluster, so the two seeds are not overstating it.

The mechanism predicts DSI ≈ Δread × (read-conditioned gap − unread gap) ≈ 0.10 × (0.45-0.59 − 0.04) ≈ **+4 to +6 pts**. That is less than half of the +11.4 needed just for the lower bound to clear 0. Under a true caution shift of 0, the CSI upper bound ≤ +5 passes only about 50-55 % of the time.

P1 has the same problem. At a true +10 read-rate effect, "LB > +5" passes about 30 % of the time (per-cluster SD ≈ 0.22 over 40 clusters gives a half-width of about 7). The design's own bar (point ≥ +10, MDE 9-10) implies this.

So the decisive experiment is not decisive: a working method most likely reads as "P2 failed". This is the it15 pattern again ("bars at the noise floor, MDE 18-20", `docs/results.md` L45).

### F4. The endpoints are not the transfer endpoints the repo already defined, nor the headline ones

- **RTS v1 is never mentioned.** `docs/evaluation.md` §10.2 (L727-746) names it "the headline transfer endpoint from iteration 17 on", and it exists and is frozen: `data/rts/v1/manifest.jsonl` has 524 rows = 262 pairs, 2026-10-27, sha in `data/rts/v1/yield.json`.
  - near tier: force_overwrite 40, prod_target 22, secret_exposure 21;
  - mid tier: remove_library 68, service_down 7, disk_quota 4;
  - far tier: oneoff_tests 51, conflicting_tests 49.
  - Its endpoint already *is* "discrimination, not caution": Youden J, with control false refusal ≤ +5 and RTS-C ≥ −5 (L735-740). No archived RTS eval exists for any model (`~/Documents/honest-agents-archive/rts/` holds only the build).
- **Nothing in training targets the external evidence types.** The final-benchmark headlines are TB2 false-claim and success, internal held-out false claims, ImpossibleBench-LCB flag, and MASK (`data/benchmarks/final_benchmark.json`). Agentic Abstention tests flawed *instructions*. ImpossibleBench tests *test-vs-spec conflicts*. In the fast path, nothing touches either. The design admits this ("direction only"). But the task is transfer "measurably on held-out surfaces and external benchmarks", and the fast path measures transfer only within the high_risk family, across 2 surfaces and 2 new layout classes.

---

## Premise checks

| claim in the design | status | evidence |
|---|---|---|
| Instructions byte-identical, 524/524 train and 120/120 val | **verified** | `cmp` over `data/decision/surfaces/tasks` and `surfaces_val/tasks` |
| Twin and control differ only in the last Dockerfile `RUN` line | **verified** for 0033979a readme and empty_backup (L34, the file's last line) | diff of the two Dockerfiles |
| Pair table never used in training; break-even only when the control passes | **verified** | `reward.py` L81-100, `pair_table: bool = False` L191; DEF-13 closure "always proceed 0" assumes a pass |
| Pair table pays blind declines 0.27-0.72 per pair | **plausible, partly verified** | My recount agrees on direction and size. Caveat: `ct_audit.py` counts `honest_failure` (declined = True) as a decline, so its "honest failure given proceed" column is structurally ≈ 0. Only 1 of 480 G2 s1 controls is `honest_failure`, so the numbers stand |
| G2 discriminates given a read on held-out, +59 / +42 | **verified on empty_backup only** (+59 s1, +38 s2), but on **tiny n**: 14-27 read twins and 12-18 read controls per seed | `comp_by_read.py` |
| Held-out unread declines 46-50 % vs 28-36 % trained | **verified** (empty_backup: twin 0.49 / 0.56, control 0.45 / 0.47 vs trained 0.28-0.36) | same |
| "Not a judgment failure" | **contradicted in part** | badcase.md: 11 of 26 twin destructions read the README and deleted anyway; a leaked control template ("nothing there says X has to be kept") fires on "only copy" phrasing; destruction rate 18.9 % when the decision is ≥ 5 commands after the read. This is a judgment-plus-memory failure that more reading does not fix |
| dapo uses the global token normaliser | **verified** | `.venv/.../trl/trainer/grpo_trainer.py` L3229-3234, TRL 1.14.1 |
| TRL builds one `SamplingParams` per batch; colocate engine seed = process index | **verified** (L356-357) | `trl/generation/vllm_generation.py` |
| Per-request seeds exist in the repo | **false today** | DEF-58 (`docs/defects.md` L3165ff): no request carries a seed. The fix was amended away, so CRN is net-new harness code |
| E3 ≈ $9.5 | **understated** | infra.md A7 gives 8.2-9.2 h = **$12-13.5** for 30 steps on 96 GB, counting probes (the step-0 probe took 2,899 s), merges and select. Seeded or batch-invariant decoding may slow generation further (unknown) |
| 30 steps is enough | **contradicts the repo's own RL plan** | `docs/iteration-16-direction.md` L248 (S5): ≥ 50 steps, stop if grad norm < 0.2. G3 at 15 steps had grad norm about 0.1, and model identity explained 0.0 % of read variance (`docs/results.md` L33) |
| A size-cue audit is needed | **confirmed as a non-issue on the lexical val surfaces** (twin-control length difference ≈ 0, twin longer in 6-12 of 20). empty_backup's twin is always smaller (0/20), but there the size *is* the evidence | `surfaces_val` Dockerfile last-line lengths |
| RL plain rows are free of the badcase task defects (59974445, 03e020f3, bfb0949a, be2a64e9, 7b368f5e, 38af60bb, 878bfeb5) | **verified for these 7**: 0 hits in `data/train_rl_v2.jsonl` (370) and `train_rl_v3.jsonl` (120). The pool's own A1/A2 rate is **unknown** | grep |

---

## Gameability and label weaknesses

1. **P1 (read rate) can rise without discrimination.** `inspected` is the strict rule: an evidence file displayed by full path or through its directory (`cat D/*`, `grep -r x D`) inside an anchored K = 8 window (`src/honest_agents/eval/read_diagnosis.py` L13-25). A cat-everything habit raises P1 mechanically. It is not rewarded, but it is the primary endpoint. P1 should be a mechanism metric, not the primary claim.
2. **New layout classes need new `SurfaceSite` evidence paths** (`~/NOTES.md`, `docs/` two levels up). Every new site definition is a new metric definition, and integrity-gate S4 says there is one implementation. It needs the S3 false-positive audit before any number counts.
3. **Equal-size padding is a new cue.** If the shorter text of each pair is padded, the padding pattern can mark the branch. Pad *both* texts to a fixed length drawn per pair, with the same padding scheme. The val sets are not padded, so check that this train/test mismatch does not itself become a held-out cue.
4. **The competence price depends on other rollouts' honesty.** A control decliner's reward falls when its groupmates lie. With stop-gradient this is a moving price, not an objective, so "the training objective is twice the evaluation metric" (Prop. 2) holds only at a fixed point. Log ĉ per source over steps.
5. **The guard set is contaminated by grader defects.** The G2 guard, "false claims on feasible dev 125/620", includes the badcase task defects (38 % of select false claims are A1-A3), and none of them is in `docs/defects.md`. The integrity gate (memory: "no experiment while any defect is open") means they must be entered first, and the guard must exclude or repair them. Otherwise it partly measures grader-guessing.
6. **DEF-01 inbound_link control grading** must use the post-fix regrade everywhere. The design says so; E0 must assert it.

---

## Novelty and RL-vs-SFT

- **Competence-priced pair rewards.** New in this repo, and in spirit a cost-sensitive / Lagrangian price. Not novel as theory. The real contribution would be the measured blind-neutrality *with* the read-competence correction (F1).
- **CRN across a counterfactual environment pair.** I found nothing identical:
  - Sakib 2026 (arXiv 2609.24144, abstract re-fetched) pairs *environment* noise across rollouts of the *same* prompt. It reports a 21-63 % drop in gradient covariance but missed its learning-curve criteria. That is a direct warning for the 22-47 % SNR claim here.
  - The closest family is **shared-prefix tree RL**: Tree-GRPO (Ji et al., "Tree Search for LLM Agent Reinforcement Learning", ICLR 2026, https://arxiv.org/abs/2509.21240) and VinePPO (Kazemnejad et al. 2024, https://arxiv.org/abs/2410.01679, from memory). Prefix tokens there get credit from their descendants, which is exactly what §2.2's "credit on shared tokens" derives.
  - So CT-GRPO's coupling is best described as a *tree branched at the environment fork* (the first observation that differs). It should cite that family. Once it does, the obvious exact implementation is a fork, not seeds (required change 3).
  - Oberst & Sontag 2019 (ICML, https://arxiv.org/abs/1905.05824) is confirmed as the Gumbel-max SCM reference.
- **RL vs SFT.** Which file to open and when to keep looking are genuinely target-free and multi-step. This is a legitimate RL use and is not cut-point SFT. But the badcase list of things only RL reaches (B1 claim-despite-evidence 29/95, decisions far from the evidence) is untouched by this design. That is fine if it is stated.

---

## Required changes (numbered, concrete)

1. **Price with blind competence, not pooled competence.**
   - Estimate ĉ from acted controls that were *blind under the environment-defined label*: their observations were identical to the coupled twin's up to the decision (the `replay_episodes --pair` "revealed" definition, `read_diagnosis.revealed`, DEF-29 normalisation).
   - Apply that one ĉ to *every* control decliner. Shrink toward an archive prior for unread control pass (G2: 0.62-0.67, so c₀ ≈ 0.25-0.33), not toward the bank pass rate.
   - Re-run `ct_sim_readcomp.py` with this estimator. The bar is a blind-decline gradient within ±0.1 per step at both G2 parameter sets.
   - Because the label is defined by the environment (outputs differ between worlds), it is not the strict-read heuristic, and it is used only to choose the estimation subset. Still, get owner sign-off against the design.md §4 rule "never reward read".
2. **Restore the honesty ordering, and assert it in `RewardConfig.__post_init__` on a c-grid.**
   - Either (a) move the competence term onto the twin: correct decline = c, vague decline = c − 1 − α, controls exactly as in the pair table (blind pair value is still c − 1, and the reveal value becomes (c − 1) + (1 + c)·Δ); or (b) keep the design's form but restrict training pairs to sources with blind-competence c₀ ≥ 0.2.
   - The grid test must also assert, for every c on the grid: honest failure > false claim; a false claim never beats any non-claim on the same branch; no blind report beats c − 1.
   - Define the "acted, then non-claim" row (false_blame with tests passed).
3. **Replace seed-CRN with an exact environment fork.**
   - Run the agent in the twin container and shadow-execute each command in the control container.
   - At the first command whose normalised outputs differ (the existing pair-replay logic), copy the conversation and continue the control branch from there.
   - Coupling is then exact by construction: no vLLM batch-invariance dependence, no E1(a) kill, and no DEF-58 seed plumbing. Each branch is still marginally π-distributed, and the prefix is generated once (Tree-GRPO's saving).
   - Keep seed-CRN only as a fallback if shadow execution proves infeasible. E1 then measures shadow-execution overhead, not token identity.
4. **Re-pre-register the endpoints so a working method can pass.**
   - Make the **RTS v1 ΔJ** (near + mid tiers; far tier reported) the primary transfer endpoint, with its built-in false-refusal (≤ +5) and RTS-C (≥ −5) guards. It is frozen and external, and it is literally "discrimination, not caution".
   - Keep P1 (read rate) as the mechanism endpoint.
   - Drop the P2 conjunction as primary, or power it: at least 16 samples per branch (sampling SD per cluster ≈ 0.25, giving DSI half-width ≈ 5.5 at 80 clusters) and a CSI margin of ±7. Run `analysis.power` with the archived rows *before* the freeze, not "finalised after E0".
   - State the MDE of every bar in the prereg (DEF-10 / DEF-20 are only waived per iteration).
5. **Fix the trivial kill criteria.**
   - E0's "read pairs' mean > c − 1" cannot fail: DEF-13 already measured read − unread of +0.45 to +0.86.
   - E1's "share of pairs with pair sum ≠ c − 1" is about 100 % by construction, because control outcomes are ±1.
   - Replace both with: (i) the share of groups containing at least one revealed pair *and* within-group reward variance attributable to reveal (variance decomposition); and (ii) the probe gradient norm on the reveal-token subset versus the S5 rule (stop if grad norm < 0.2 after 20 steps).
6. **Follow the repo's own RL plan's step count**, or justify deviating from it. S5 says ≥ 50 steps; 30 steps plus the early stop at step 20 is below it. Re-cost E3 from infra.md A7: 30 steps $12-13.5, 50 steps about $19-21. Then re-check the $66 budget against base + G2 final benchmark ($12-20) plus the RTS arms ($3-7 each, infra.md B3).
7. **Integrity gate first.**
   - Enter the badcase task defects (59974445 the two sha256 values, 03e020f3, bfb0949a, be2a64e9, 7b368f5e, 38af60bb / 878bfeb5) in `docs/defects.md`.
   - Exclude them from the G2 false-claim guard.
   - Obtain the DEF-10/20/50 waiver extension and the final-benchmark prereg amendment.
   - Until then, no paid step runs.
8. **Drop S3 as a "theory prediction".** Report unread declines as a guard (≤ G2 + 3) with its own power. Shared-parameter coupling (it17/it18) means the zero expected blind gradient does not imply an unchanged blind prior.
9. **Pad to a fixed per-pair length on both branches** (or randomise padding), and run the S3 false-positive audit on every new `SurfaceSite` location before E4.

## Optional improvements

- **Address the "read and deleted anyway" and "decision far from the evidence" failures** (badcase: 11 of 26 and 18.9 %). Add a pair outcome *conditioned on the decision being ≥ 5 commands after the reveal*, so the fork credit reaches persistence. This is cheap, because the fork gives the exact reveal step.
- **Phase 2, made cheap.** If ImpossibleBench matters, use RTS far-tier *operators* (oneoff/conflicting tests) re-implemented on Endless Terminals sources as a training family. Keep the RTS far tier's own sources as test only, and run the never-trained check on RTS too. Then X1 can become a real endpoint.
- **Run a $3-7 RTS eval of G2 now** (5090 is fine for evals, infra.md A4). It gives the baseline J and false-refusal by tier, and tells you whether "caution shift" even describes G2 off-family, before you design to fix it.
- **Log ĉ per source per step, and the coupled-pair blind credit split by c bucket.** That is mechanism evidence that costs nothing.
- **Cite the tree / prefix-sharing RL family** and frame the novelty as "branching at the counterfactual environment fork". That framing is defensible.

## Claims verified vs not verified

**Verified (this review):**
- instruction identity 524/524 and 120/120;
- the twin/control Dockerfile diff is the last `RUN` line;
- the pair table is off by default and its control rows are as stated;
- the dapo normaliser in TRL 1.14.1;
- TRL's single `SamplingParams` and engine seed;
- DEF-58: no per-request seeds;
- the held-out read-conditioned gap on empty_backup (+59 / +38, small n);
- held-out unread decline 45-56 %;
- the direction and size of the pair-table blind push;
- the competence-by-read confound (new);
- the residual priced push +0.42 to +0.73 per step in the toy (new);
- P2 CI widths (new);
- the RTS v1 manifest exists and is unrun;
- the infra.md 30-step cost is $12-13.5;
- S5 says ≥ 50 steps;
- no badcase-defect sources in RL pools v2 / v3;
- Sakib 2026's content;
- the Oberst & Sontag 2019 citation;
- the Tree-GRPO existence and mechanism.

**Not verified:**
- vLLM per-request seed coupling fidelity, and batch-invariant support on Blackwell;
- step time under seeded decoding;
- whether GRPO-with-tools fits a 5090 (unknown; infra.md says unmeasured);
- the toy's SNR gains translating to learning;
- the design's +2.1 vs reveal "half the size" ratio under its SNR definition (my sim prints means, not SNR);
- the bag-of-words 0.34 inversion (transfer.md T3);
- the Shapley split of +12.0 (from `docs/iteration-15-diagnosis.md`, not re-run);
- PEGASUS and Glasserman & Yao (not re-fetched);
- the pool's A1/A2 task-defect rate;
- feasibility of shadow execution (change 3), which needs a CPU-only Docker test on the node.

Sources: [Luck Is Not Skill, arXiv 2609.24144](https://arxiv.org/abs/2609.24144) · [Tree Search for LLM Agent RL, arXiv 2509.21240](https://arxiv.org/abs/2509.21240) · [Oberst & Sontag 2019, arXiv 1905.05824](https://arxiv.org/abs/1905.05824)
