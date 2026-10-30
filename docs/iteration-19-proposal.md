# Iteration 19 proposal: close the gaps (DRAFT, for the owner)

The owner's framing (2026-10-29): the contribution is a training strategy that optimises task success under honesty
constraints that work against it, with RL covering what SFT cannot, judged by the final benchmark's headline metrics.
The work was planned by an 8-agent research pass: four research reports, two designs, two adversarial reviews. The
reports are in [iteration-19-research/](iteration-19-research/). Their helper scripts were scratch files and are not
kept. Nothing was run on a GPU, in Docker or against a paid API.

## 1. What the research found

**Bad-case analysis of G2's false claims** ([badcase.md](iteration-19-research/badcase.md); 95 failing claims on the
select split read in full, 1,678 pool claims counted):

| cause of a false "complete" claim (select, n = 95) | share | who can fix it |
|---|---|---|
| **task defects:** the instruction contradicts itself, or the test checks something never stated (7 sources) | **38 %** | nobody: fix or drop the tasks |
| G2's own check showed the failure and it claimed anyway | 31 % | RL (the evidence is in context) |
| a spec detail was missed; the output was never compared to the spec | 21 % | RL plus capability |
| blind check, fabricated step, zero work | 9 % | rules or RL |

- **Checking is not the problem.** 92 % of claims, passing or failing, check after the last write. Failing claims are
  near misses (median 80 % of tests passed).
- **Struggle predicts failure.** More commands, self-doubt in the prose and rewrites predict failure (AUROC about
  0.70), and G2 does not act on it.
- **Half the false claims are beyond G2's skill.** About 45 % sit on variants G2 never solves in 4 tries, and it still
  claims on two thirds of those.
- **Twins.** The remaining twin destructions come from a lexical rule leaked by the iteration-14 SFT template
  (*"nothing there says X has to be kept, so it is safe"*). They also come from deleting long after the read: 18.9 %
  destroyed with 5 or more commands between read and delete, against 2.3 % right after the read.
- **remove_input controls.** 77 % of the workaround claims never disclose the missing input.

**Final benchmark** ([infra.md](iteration-19-research/infra.md) part B):
- Never run.
- Running base and G2 now needs:
  - a prereg amendment: it16 prereg 10.3/10.4 forbid this order;
  - waivers for a new chain name;
  - redefining `internal_heldout` to the frozen test split, since its build does not exist;
  - a pinned eval code path, so a later third arm is comparable.
- Cost:
  - **core plan** (TB2 S=4, ImpossibleBench-LCB S=4, internal held-out S=2, MASK generation only): **$12.5-20 at
    $1.11/h**;
  - **lean plan:** $8-13.
- **Caveat:** TB2 success for a 4B model is near zero (0-4 of 89 in the iteration-14 check), so TB2 false_claim must be
  reported with the claim rate.

**Constrained RL design, CR-GRPO** ([design](iteration-19-research/design_constrained_rl.md),
[review](iteration-19-research/grill_constrained_rl.md)). The idea:
- full-episode GRPO from G2;
- an exact counterfactual gradient on the claim/not-claim decision, which plain GRPO cannot move: 72 % of false claims
  are on variants claimed 4 of 4 times;
- Lagrangian constraints on false claims, twin damage and control refusal;
- a "caution-neutral" pair table, so blind declining earns nothing.

**Review verdict: GO WITH CHANGES for the $0 steps, NO-GO for any paid step as written.** The fatal flaws:
1. its small GPU kill test would have passed it18;
2. its claim threshold breaks its own success guard;
3. joint power is about 0.3;
4. it lowers the twin-destruction stake about 3x;
5. most of its "action credit" equals plain GRPO in expectation.

The review lists 11 required changes. What is new and holds up is the state-graded counterfactual on the report
decision, and the caution-neutral constrained pairs.

**Transfer design, CT-GRPO** ([design](iteration-19-research/design_transfer.md),
[review](iteration-19-research/grill_transfer.md)). **Review verdict: NO-GO for the paid run.**
- Its central theorem ("zero caution gradient") fails on G2's data, because reading predicts control competence.
- Its priced reward ranks a false claim above refusing on a control.
- Its primary endpoint cannot pass at the effect size it predicts.

The diagnosis stands: on new layouts G2 doesn't reach the evidence, declines blind more often, and some evidence is
relational. The fix is not ready.

## 2. Proposed plan, in order

1. **Integrity, $0, now.**
   - File the 7 defective sources in `docs/defects.md`. 38 % of false claims on the select split are task defects,
     so the metric partly measures broken tasks.
   - Audit the pool for the same defect: does every literal the test asserts appear in the instruction?
   - Record the leaked SFT template as a defect of the iteration-14 data.
2. **Headline baseline, about $12-20.** Run base and G2 on the final benchmark, core plan, on a $1.11/h-class node.
   - First a short amendment to the it16 prereg (order, test-split held-out, pinned eval path) and waivers for the new
     chain.
   - This gives the first public-benchmark numbers for the project and the baseline every later claim needs. It is
     worth doing whatever the RL shows.
3. **CR-GRPO, revised with all 11 required changes. $0 checks, then a kill test of about $5-7.**
   - The kill test includes a real rollout on the 120 twins, so it can catch an it18-style regression.
   - The full run (about $14-21) happens only if that passes.
   - The scope is stated honestly: the report-decision term is supervised learning on the policy's own states with
     outcome labels. The contribution is the joint on-policy system, shown against each of its parts.
4. **Not now:**
   - **CT-GRPO.** Its valid parts, competence pricing that accounts for reading and twin/control common random numbers,
     go into a later revision.
   - **A "conflicting visible tests" family.** It is the cleanest RL-only target for cheating under pressure, but it
     costs 1-2 days of authoring.

## 3. Budget

The balance is $74.07. With the G3 rule's $25 floor that leaves $49 usable; with the watcher's $8 floor, $66. Steps
2 + 3 come to about $31-48. The order above puts the baseline first, because it is useful under every outcome.

## 4. Owner decisions needed

- The plan and its order (integrity → baseline → revised CR-GRPO).
- For step 2: the prereg amendment, the waivers for the new chain name, and the GPU spend, asked separately when the
  amendment is written.
- The budget floor: $25 or $8.
- Whether MASK's paid judge is ever wanted. Without it, MASK responses are generated and frozen, and scored later.
