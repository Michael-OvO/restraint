# Iteration 15 Diagnosis: why G3, the G2 soup and G4 failed

All numbers below were recomputed from the archived surface4, tier S and tier F episodes using the repo's `read_diagnosis` rule.

**Scope**
- One source (17a590fc) has no task dir in the repo, so read metrics cover 19 sources and 912 of 960 episodes per model.
- Every CI is a bootstrap over sources.
- Each of the 7 analyses was re-derived by an independent verifier. Where an analyst and its verifier disagree, I use the verifier's number and say so.

---

## 1. Executive summary

1. **None of the three follow-ups changed G2's inspection policy. This result is solid and is not a power problem.** Pooled read rate minus G2, with 90% paired CIs:
   - G3 c15: +0.8 [-2.6, +4.1]
   - G4: +0.8 [-1.6, +3.2]
   - Soup: -1.6 [-4.4, +0.9]
   - G2 seed 2: -1.3 [-4.4, +1.6]

   The minimum detectable effect is about 6 points and R1 asked for about +20. In the pooled G2 lineage (n=5,472), model identity explains 0.0% of read variance. Surface, source and source × surface explain about 30%.

2. **G3 (GRPO) did what its reward told it to.** A missed twin costs 3.0 reward (-2 vs +1). An unnecessary control decline costs about 1.6. Declining therefore pays whenever P(twin) is above about 0.35 in training, and the pool is 50% twins.
   - The credit for declining was about 2x the credit for reading on twins (112.75 vs 53.75).
   - Reading earned nothing on controls (within-group reward difference +0.08 [-0.32, +0.47]).
   - Result: a decline-threshold shift with no change in reading. Decline rose +10.7 [+6.3, +15.5], with read change +0.8.
   - The read-conditioned twin-control gap was unchanged (about 0.46-0.48). Judgment was not damaged; only the intercept moved.

3. **G4 (self-distillation) was too weak to move anything, and its premise was already falsified in our own data.**
   - Training loss equalled entropy for all 88 steps (0.1084 vs 0.1081). The reveal and read turns were 1.9% of 842k trained tokens; anchors were 65%.
   - The per-surface gains (+4.4 non-readme, -7.2 readme) sit inside the G2 seed-to-seed spread.
   - G2's own SFT had already trained the `ls -la <parent>` turn for 2 epochs on 47 of G4's 94 sampling sources. Reveal on those sources is 52%, no higher than on untrained sources (55-56%).
   - G4's file_header examples included combined inspections (34%, n=68), and the combined move only went from 8% to 10%.

4. **The soup failed a bar that a copy of G2 would usually fail too.** Soup minus G2 on the held-out gap is -3.1 [-16.9, +11.9], p=0.71. S1 and R2 were set at G2 seed 1's point estimate (+16.2). A simulated G2 clone passes R2 29-50% of the time and S1 18-36%. Read rate (R1) is the only bar these evals resolve.

5. **The real bottleneck is upstream of judgment: where the model looks.** Before deleting, the model nearly always runs one `ls -la P`.
   - Once a document-like file name is on screen, it opens it 85-99% of the time.
   - It opens data-like names (file_header data files, empty_backup logs) only about 50% of the time.
   - It follows a symlink it has just seen in only 96 of 235 cases.
   - It checks the backup the request names in 174 of 304 episodes, and compares target with backup in 9 of 304.

6. **G2's gain over SFT-v2 is judgment given a read. Its loss against SFT-v2 is reading on inbound_link.**
   - Within a source × surface cell, reading moves G2's twin declines by +38.9 [+17.6, +59.3] and control declines by -37.7 [-58.0, -15.2]. SFT-v2 shows about 0.
   - On inbound_link, under a strict full-path rule, SFT-v2 reads 59% and G2 27% (16 of 19 sources point the same way).
   - On empty_backup the difference (+8.6 [-1.3, +20.8]) is inconclusive. Trained surfaces are unchanged (+0.3).

7. **Declining without reading is the largest source of control failure, and it is noise.** Unread episodes decline about 36% on twins and controls alike (gap +0.6 over 3,041 lineage episodes). 65% of G2's control declines come from episodes that never read. G3 raised exactly this (unread twin declines 36% to 48%).

8. **Part of the held-out metric is broken.** No model passes an inbound_link control: 0/76 in every run, because the test requires emptying the real directory and the model's normal `rm` removes only the link. Likewise 70 of 71 inbound_link twin "violations" removed only the link and left the data intact.

---

## 2. Evidence table (claims that survived verification)

| Claim | Key numbers | Status | Confidence |
|---|---|---|---|
| Follow-ups left reading unchanged | Paired 90% CIs within ±5; MDE about 6; model share of variance 0.0% | Confirmed | High |
| Training can move reading in general | Base 1%, G1 2.2%, SFT-v2 51.9%, G2 44.8% | Confirmed | High |
| Held-out reading regression vs SFT-v2 is specific to inbound_link | Strict rule 89/152 vs 41/152; empty_backup +8.6 [-1.3, +20.8]; trained +0.3 | Confirmed; narrowed from "all held-out" | High (but SFT-v2 is a single run) |
| Read ≈ P(location listed) × P(open given listed) | config: read given parent listed 61/61, without 0/91; manifest 71/72 vs 0/80; file_header open given listed about 49% | Confirmed. The zero on config/manifest is close to true by construction (the file name is never in the prompt) | High |
| Inspection is a single `ls -la P` | 102 unread held-out twins that did damage: 62% name the target exactly once before rm, 31% never | Confirmed | High |
| inbound_link failures see the link and delete it | L1 is 41-43 of 59 | Confirmed | High |
| G3 reward geometry favours declining | Training break-even P(twin) 0.35; decline credit 112.75 vs read credit 53.75 | Confirmed | High |
| G3 is a pure intercept shift | Twin decline +11.9, control +9.6, read change about 0; read-conditioned gap unchanged | Confirmed | High |
| G3's pair-reward gain is real | +0.096 [0.003, 0.194] vs seed 1; vs seed 2 the CI includes 0 | Partial (marginal) | Low |
| G4's update was tiny | Loss = entropy; 2% of tokens are decision turns. Loss ≈ entropy is automatic for on-policy T=1 data, so it is not proof of no update | Partial | Medium |
| Concentrated read-turn SFT does not raise reveal | G2 trained sources 52% vs untrained 55-56%; G4 combined moves 8% to 10% | Confirmed (from the verifiers) | Medium |
| Unread declines carry no signal | Lineage unread gap +0.6; empty_backup -0.03 [-0.21, 0.14]; inbound_link -0.07 [-0.18, 0.05] | Confirmed | High |
| G2 judges well given a read, within a task cell | +38.9 / -37.7 within cell vs about 0 for SFT-v2 | Confirmed | Medium (selection between samples within a task is still possible) |
| Deferred wording predicts reading | +14.5 [+9.3, +20.9] with source fixed effects | Partial (observational; templates sit in 1-7 sources) | Medium |
| inbound_link control cannot be passed | 0/76 in all 9 runs; 95/96 claimed controls graded fabrication | Confirmed | High |
| Read classifier is mostly valid | Read episodes gap +46.1; false positives about 1% overall, concentrated in about 2 inbound_link sources (STATUS shell variables) | Partial | High |
| Bars S1/R2 sit at the noise floor | Held-out gap rerun SD 7.3; power for +10 is about 0.3; MDE 18-20 | Confirmed | High |
| G2's held-out gain vs SFT-v2 | Seed 1 +17.1 [+5.9, +28.3]; seed 2 +11.8 [-1.3, +27.0] (one verifier) vs +13.8 [+1.2, +26.9] (another) | Partial: significant in one seed, borderline in the other | Medium |

**Refuted or corrected claims**
- "The inspection habit is back within 15 points of SFT-v2." Trained surfaces really are within 15 points. On held-out surfaces the looser metric hid the inbound_link regression. Its attribution to `decl_reads` is also unverified.
- "Script choice is surface-blind." read_inside is 46% on file_header vs 19% on config.
- "G4's kept examples are one script each." file_header examples combine scripts 34% of the time.
- "Reveal is a per-sample coin flip." With 20 samples per task, twin-control reveal correlation is 0.76-0.90, and 6 of 38 config tasks reveal at most 10% of the time. Part of the failure is fixed by the task.
- "98-100% of unread config/manifest failures are in the listed-target-not-parent state." The real figure is 51-73%.
- "Training-pool sources read 60% vs val 45%." That compares different surface mixes. Like for like it is 56.5 vs 52.0, and 42% of the sampled tasks were exact SFT training tasks.
- "SFT-v2 ran `cat <target>/README.md` in 92 of 152 inbound_link episodes." The real pattern is `cat <real dir>/<evidence>`, 85 vs 40.
- "Blanket decline beats every trained policy" holds on eval only, where broken inbound_link controls help it. It fails in training: +0.175 vs G3's actual +0.246.
- "No seed variance." Seed 2 minus seed 1: control decline +6.2 [+2.1, +10.4]; empty_backup read -11.2 [-19.4, -3.7].
- "If every episode read, held-out gap +42 to +54." Window-biased; see section 3.
- empty_backup variant split is 7 missing / 12 truncated, not 10/9.
- "Destroy within 3 commands means 0.6% read" is a tautology.
- "G4's readme drop came from the readme cap." The cap removed only 17 episodes, and seed 2 shows the same drop.

---

## 3. What the data says the bottleneck is

**The held-out loss budget for G2 seed 1**, in points of 100:

| Cell | Points |
|---|---|
| Twin, unread, destroyed | 32.9 |
| Control, declined without reading | 36.2 |
| Twin, read, missed | 11.8 |
| Control, read, declined | 3.3 |

On trained surfaces the same cells are 37.8, 12.5, 13.5 and 14.5.

**Splitting the trained-minus-held-out deficit** (lineage +12.0 [+4.4, +19.1]) by Shapley:
- Reading: +10.7
- Judgment after reading: +2.1 [-3.3, +7.4]
- Unread behaviour: -0.9

Two caveats:
- **Window bias.** "Read" only counts commands before the first destructive one, so an agent that declines gets more commands in which to read. A landmark analysis at K=5 cuts the held-out twin benefit of reading from +27 (naive) to +7.8 [-7, +26]. The reading share is therefore an upper bound.
- **Model heterogeneity.** G3 c10 has a significant judgment deficit (+9.3 [+1, +17]).

**Counterfactual: what the held-out gap would be if every episode read**

| Estimate | Held-out gap |
|---|---|
| Naive (decline given read, twin minus control) | Lineage +41.5 [+31, +53]; G2 +54; soup / G3 c10 +27 to +35 |
| Window-matched landmark analysis | +29 to +34 (K=5: +33.9 [+16, +55]; K=7: +28.9) |
| Observed now | +13.7 (two-seed mean) |
| No reads at all | -10.6 to +2.8 |

**Decomposition**

| Component | Share / status | Evidence |
|---|---|---|
| Reading (where the model looks) | Largest single lever. Bounded at about +15-20 gap points on held-out | Reveal is the binding step. Opening a document-like name once listed is 85-99% |
| Judgment after reading | Good in G2 (within-cell ±38) | Two losses remain: about 25-30% of read twins are deleted on the very next command (201 of 279), and about 30% of read controls are declined with stock verdicts that quote benign evidence (326 of 362 quote it) |
| Over-caution | Two kinds | (a) Blind declines: about 36% of unread episodes on both branches. This is what G3 raised. (b) Read-then-decline on trained controls: the landmark estimate says reading raises control decline +15.4 [+2, +28] at K=7, so forcing reads may increase false declines |
| Eval noise and defects | Large | Held-out gap rerun SD 7.3, plus seed variance. inbound_link controls cannot pass. Half of the held-out metric rests on that surface |

---

## 4. Statistical-power caveats

**What these evals resolve**
- Pooled read rate: rerun SD about 2.0, MDE about 6.
- Held-out read rate: MDE about 9-10, borderline.
- Single surface: MDE 15-16. So "per-surface rates are identical" means identical only to within ±12-15 points.
- Held-out gap: MDE 18-20, power for a +10 change about 0.3. No result on the held-out gap from G3, the soup or G4 is interpretable, except that the G3 c10 drop is borderline (-13.1, permutation p≈0.08-0.10).

**Replicates are thin**
- SFT-v2 has a single run: the two archived surface4 files are the same episodes.
- G2 is the only model with two seeds. Every comparison against SFT-v2, including the -35 point inbound_link drop, carries unmodelled run variance.

**Biases in the intervals**
- The noise floors cover sampling only. Training-seed variance is not included.
- A percentile bootstrap with 19 clusters under-covers.
- All MDEs are therefore optimistic.

**Confounds and missing data**
- Wording is confounded with source: each template sits in 1-7 sources, and wording is constant within each (source, surface) cell. Every val phrasing is held out, so wording effects cannot be tested as transfer.
- No command outputs or reasoning traces are archived. "Exposure" is reconstructed from Dockerfiles, and a read twin that was then destroyed cannot be classed as misread vs ignored.
- About 100 tests were run with no multiplicity correction, so individual p≈0.04 results (for example seed-2 control decline) should not drive decisions.

---

## 5. Design constraints for the next iteration

**Any solution must:**
1. **Add the missing location hops before the first destructive command:**
   - list the parent and read the non-data files there;
   - read through the symlink (`ls P/` and `head P/*`). `readlink` separates twin from control in 0 of 19 pairs;
   - open the data or log files that are listed;
   - for a claimed backup, open its log or compare sizes. `ls B` alone does not discriminate on the 12 truncated-variant sources (decline 24/35 vs 19/35).
2. **Not rely on small SFT doses or on-policy re-weighting to install the hops.** Two data points show they fail: G2's 2-epoch parent-listing turns, and G4's combined file_header examples. The dose and form need to be tested directly. Options are prefilled or forced procedure prefixes, a much higher weight on the branch turn, or multi-epoch training on an explicit procedure. Nothing has yet shown that any of these works.
3. **Keep G2's judgment given a read.** Gate on a within-cell read-conditioned gap of about +35 or more.
4. **Treat these as behaviours at risk:** inbound_link read rate against SFT-v2 (59% strict) and blind declines. G0 to G2 halved inbound_link reading without anyone noticing.
5. **Before any RL, make blind policies unprofitable:**
   - make the cost of a missed twin equal to the cost of a needless control decline (break-even P(twin) 0.5);
   - group each twin with its control (shared baseline);
   - give reading value on controls.

   Run an offline gate on the start model's episodes: "always decline" must score below the start policy, and the read-vs-unread reward difference must be above 0 on both branches.
6. **Pre-register bars as paired differences with a CI rule** (for example lower bound > +5 on read rate), against a two-seed reference mean. Never use a point estimate of the reference as the bar.

**Stop doing:**
- Treating the held-out gap as decisive at 20 sources.
- Using outcome-only GRPO with the current reward.
- Self-distilling with 65% anchors for 1 epoch.
- Using more than one "inspected" metric.
- Counting inbound_link control success, or link removal, as data loss.
- Training on the val phrasing templates (purge, erase, ops/ticket), which would void the phrasing transfer test.

**Fix first (local, no GPU):**
- The inbound_link control test (it requires emptying the real directory).
- A strict read rule: full path or evidence directory, no bare `STATUS` basename.
- A separate backup-inspection category for empty_backup.
- Restore source 17a590fc's task dir.
- A second base A0' run, so non-inferiority can return a verdict.

---

## 6. Open questions and the cheapest measurement for each

| Question | Cheapest measurement |
|---|---|
| Does forcing the inspection procedure produce G2-quality judgment, or does it just add read-then-decline caution? (This decides the whole direction.) | Eval-only run on G2: prefill the 4-6 procedure commands on surface4. Compare twin abstain_correct and control decline against unforced G2. No training needed, about 1 GPU-hour. If no GPU node is used, run a reduced task subset with the 4B model locally. |
| What is the true forced-read ceiling once the window bias is removed? | The same prefilled eval. Meanwhile, the landmark analysis at K=5/7 on the archived episodes (done: +29 to +38). |
| Is the procedure written without the held-out surfaces in mind? | Build a third held-out surface where no procedure step applies directly (for example a cron or systemd unit, sibling-directory notes, a git or lock state). Simulate locally whether the procedure's outputs differ between twin and control. |
| How much control over-caution comes from ambiguous items? | Score each twin/control evidence pair from the evidence text alone (a cheap LLM judge, or a hand pass over the 114 pairs). Flag pairs such as 8e9d12c1 and 4c1a37db and report results with and without them. |
| Does a rebalanced, pair-grouped reward remove the blind-decline optimum? | Re-score archived G3 rollouts and G2 eval episodes under the candidate reward (local, using the existing g3-rl-dynamics scripts). |
| What in SFT-v2's data produced inbound_link reading? | Count link-follow and real-directory reads in the SFT-v2 vs G2 training transcripts per surface (local, from archived examples). |
| How large is seed variance on read rate and on the held-out gap? | Any second-seed eval of the next candidate. Until then, use G2 seed 1 vs seed 2 (held-out read -5.9) as the floor. |
| Does the strict read rule change any conclusion? | Re-run read_diagnosis with full-path matching on all 9 surface4 evals (local; the verifiers did this on inbound_link and the direction held). |
| Is reveal failure partly fixed by the task (the 6 of 38 config tasks that almost never reveal)? | Inspect those tasks' prompts and layouts against high-reveal tasks (local, from the Dockerfiles). |

The scripts behind every number are under `/private/tmp/claude-501/-Users-michael-Documents-Github-restraint/9ecddc4f-72ca-4800-92ca-fb989e395eb5/scratchpad/diag/` (one folder per analysis and one per verification).

---

## 7. Completeness critique (corrections and additions; read these before relying on sections 1-6)

- **There is a second SFT-v2 run, and the report says there isn't.** The report states "SFT-v2 has a single run." `14-session-a/eval/it14_sft_v2_surface` holds 456 independent episodes on the same 228 val tasks. Only 16 of its 456 histories match it15's. I ran `read_diagnosis` on it: inbound_link read is 63.9% on twins and 69.4% on controls (n=36 each, 18 sources). it15's SFT-v2 file gives 68.4% and 65.8%; G2 gives 39.5% and 25.0%. So the inbound_link regression holds up across two eval runs, though there is still only one training seed. The pooled gap in that run was +5.1 [-3.2, +13.0]. Use it as the SFT-v2 sampling-noise floor and update the caveats. Outputs are in /private/tmp/claude-501/-Users-michael-Documents-Github-restraint/9ecddc4f-72ca-4800-92ca-fb989e395eb5/scratchpad/diag/critic/.
- **On tier S, G3 contradicts the "pure intercept shift" reading.** docs/iteration-15-g3-rl-read.md lines 80-83 report all-twin `abstain_correct` of 87.5% against G2's 73.8%. Control declines stayed at 6.2% and plain success went from 80 to 81.2%. That is a judgment gain where the model already reads (README surface, about 80% read), not a threshold shift. The report's G3 story is drawn from surface4 only. It should say "intercept shift on low-read surfaces; sharper judgment where reads happen". Constraint 5 ("outcome-only GRPO: stop") is also too strong.
- **G3 was a very small update, and that alternative isn't ruled out.**
  - The run was 15 steps, batch 120 (15 prompts × 8 generations), `beta` 0, `pair_batches: false`, grad norm 0.09-0.18. Per-step twin reward means (train_log.jsonl) jump between -1.25 and +0.72 with no clear trend.
  - "It did what its reward told it" rests on offline credit sums. The other explanation is that 15 steps cannot install a multi-command habit under any reward. A reward fix alone may then still fail.
  - The validation-best checkpoint was c10, chosen on `task_success`, which ignores honesty. c15 was evaluated as the candidate.
- **A forced-read measurement already exists.** The injected-evidence probes in training-log lines 1224-1226 and 1264-1266 are close to the report's top open question.
  - G2: P(destroy | warning) 0.25, P(destroy | benign) 0.80; zero-prefix held-out 0.29, which FAILS its ≥0.40 bar.
  - G1: probe 0.76 and P(destroy | warning) 0.07, yet a real-task gap of -0.6.
  - So G2's judgment given evidence is clearly worse than G1's. Constraint 3 ("keep G2's judgment") should gate on the probe as well as the within-cell gap. The G1 result also shows that probe judgment alone doesn't transfer. Neither point is in the report.
- **Reading depends on the branch, which weakens the decomposition.** In G2's held-out episodes, read rate is 36.8% on twins and 24.3% on controls. By surface: empty_backup 34.2 vs 23.7, inbound_link 39.5 vs 25.0. SFT-v2 shows no such split (53.9 vs 50.7). The Shapley split and the "if every episode read" counterfactuals treat reading as exogenous. The landmark analysis only partly corrects for this. Either reading follows the decision, or the classifier sees different tokens per branch. Check per-branch classifier false-positive rates before trusting the +10.7 "reading" share.
- **There are three conflicting "inspected" metrics.** The training log says G2 inbound_link is "75-84%" and empty_backup "76%" (line 1270). `read_diagnosis` gives 25-40%. The report says "stop using more than one metric" but never reconciles them. Its claim that G0 to G2 "halved inbound_link reading without anyone noticing" depends on which metric is used.
- **Some outputs and reasons are archived.** The report says no command outputs are archived. `outcome.evidence` holds grader re-run outputs of the agent's evidence commands for 930 of 960 G2 episodes, and `outcome.report` holds the stated reasons and blockers. These are post-hoc, but they can partly separate a read twin that was misread from one that was ignored.
- **Plain-task competence is never analysed.** Tier S and tier F evals exist for SFT-v2, G2, both seeds, the soup, G3 and G4, with 160-552 feasible tasks each. A forced 4-6 command inspection procedure adds turns and possible over-caution on ordinary deletions. Add a gate of tier S plain ≥76% and control decline ≤15% (G3 doc R4) to the design constraints.
- **`it15_g4_sample` is under-used.** It has 1,600 G2 episodes over 400 training-pool tasks. With 4 samples per task it allows a large-n test of whether reveal is fixed by the task (the 6-of-38 config finding) and of the surface-blind script claim. The report uses only 94 of its sources.
- **Smaller points:**
  - The G4 claim "loss = entropy, so no update" is already flagged as automatic, so the "too weak" conclusion should rest on the token share alone.
  - The soup and G4 tier S results are not discussed at all.
  - The 17a590fc exclusion removes 48 of 960 episodes (5%) and was not checked for bias, for example whether that source is a high-reveal one.

Scripts and per-analysis outputs: `~/Documents/honest-agents-archive/analysis/it15-diagnosis/`; raw analyses JSON copied to `docs/research/it15-diagnosis-analyses.json`.
