## Fact-check

I re-read primary sources for 14 claims a decision rests on; nothing was run on a GPU. Most hold as reported, five need correcting.

**Confirmed as reported**
- **verl agent loop:** `AgentLoopBase.run(sampling_params, **kwargs) -> AgentLoopOutput` is the extension point (`verl/experimental/agent_loop/agent_loop.py:466`). All `rollout.n` sessions of a prompt are asyncio tasks in one worker, and an empty list is not stored (`verl/trainer/ppo/v1/agent_loop_tq.py`, `_run_prompt` and `_agent_loop_postprocess`). v0.9.1 behaves the same.
- **Tool lifecycle:** stock `ToolAgentLoop._call_tool` runs create, execute, release on every call, and `calc_reward` is never called. A per-episode sandbox needs our own loop.
- **Std flag:** `algorithm.norm_adv_by_std_in_grpo` defaults to True and is read by the V1 trainer (`trainer_base.py:1782`). False gives reward minus group mean. A one-member group gets its raw reward as advantage; TRL gives 0.
- **verl LoRA:** `model.lora.merge` defaults to False, so only adapter tensors are synced. Any non-`naive` backend drops the adapter (`engine_workers.py:758-766`; issue #7495 open).
- **Pins:** vLLM 0.29.0 on main, 0.24.0 at v0.9.1; the 0.30.0 upgrade PR #8072 is open.
- **Qwen3 and Qwen3.5:** token builders, `verl/models/transformers/qwen3_5.py` and the example scripts exist.
- **vLLM 0.30.0:** `/v1/load_lora_adapter` exists behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING`. Generation-config defaults apply only when a request field is None, and TRL sends every field explicitly.
- **TRL 1.14.1:** the sync trainer always merges and pushes full weights. Adapter-only sync exists only in `AsyncGRPOTrainer`, which hard-codes std normalisation; PR #6007 is open.
- **Tail-chunk bug:** I re-ran the CPU simulation and it reproduces (final chunk 0% merged, 100% with a clone). verl's own code comments describe the same bug class (`transformer_impl.py:1010-1013`, PR #7014). Still not reproduced on a GPU.

**Corrections**
1. **Qwen3.5 + LoRA + GRPO is covered in verl CI.** The researcher said no shipped example combines them. A CI job runs Qwen3.5-2B, LoRA rank 32, adapter mode, V1 sync (`.github/workflows/e2e_ppo_trainer_megatron_vllm_2.yml:339-348`). It is single-turn and I did not check its pass history.
2. **The TMax recipe needs a port on v0.9.1 too, not only on main.** Both already use `ct_merge_context_msg`. Pinning v0.9.1 is therefore viable for the API.
3. **The own-trainer sketch should not use `load_inplace: true`.** vLLM then re-reads the adapter on every scheduled request, and the prefix cache keys on the adapter name only. Use a new name per step.
4. **verl's log-probability check needs `bypass_mode=False`.** The `training/rollout_probs_diff_*` metrics are skipped in bypass mode, which the researcher's command line uses.
5. **Group masking in verl shrinks the batch.** A group whose sessions all return `[]` is "finished" with zero rows and is not refilled. A step where every selected group is empty raises (`replay_buffer.py`, `sample`).

**Effort assumptions**
- `HonestGRPOTrainer` still does not exist, and `configs/base.yaml` still has `scale_rewards: group` and `num_processes: 2` (read about 18:00).
- TRL has a public `rollout_func` hook (`grpo_trainer.py:262-270, 2258-2276`) as an alternative to copying the private loop.
- Our venv has no ray, tensordict, hydra or TransferQueue, and verl's uv environments are Linux-only. The "50–60% Mac-testable" figure for verl is UNVERIFIED.

---

## Decision memo: can verl or vLLM run our training job? (2026-10-04)

**Answer.** verl can, and it removes most of our TRL workarounds. vLLM cannot by itself: it is the sampler and has no loss or optimizer, so "customising with vLLM" means writing our own trainer around it. Switching now saves no GPU money, because the next two paid stages need no custom multi-turn loop. Every day and cost figure below is an estimate.

### Options

| | A. TRL 1.14.1 as planned | B. verl | C. Own trainer on vLLM | D. SkyRL, custom generator |
|---|---|---|---|---|
| Known problems removed | Serial tools, by our own loop (not written). Std scaling, by config. Masking is done. One trainer process and merge + full push stay. | Serial tools, one-process limit, std scaling (`algorithm.norm_adv_by_std_in_grpo=False`), merge (adapter sync is the default), token fidelity, per-token advantages. We write the loop, group mask, one-member-group guard and training metrics. | All, by construction, including alternative reports. One trainer GPU. | Concurrency, std flag, adapter sync. Its Harbor generator runs Harbor's agent and verifier, so we write our own. |
| Days to a first correct full-loop step | 3–5 | 6–10; plan for the top | 6–10 | Not designed; UNVERIFIED, probably like B |
| Testable without a GPU | About 75%, on the Mac | 50–60% claimed; needs a Linux box | About 85–90% | Unknown |
| Main risks | Sync fidelity (below); memory on one GPU at 16–32k tokens; copying a private 240-line method | Our combination (sync mode, LoRA adapter, custom loop, Qwen3.5-4B, Docker) is untested by anyone; alpha API; vLLM 0.29.0, torch 2.13, CUDA 13; open report of garbled rollouts after weight sync (#7904) | Silent template or log-probability bugs; we own checkpointing and memory handling | Open report of silent logit errors with 4B LoRA when colocated (#2215); June Harbor pin |
| Paper credibility | Fine if the custom loop is disclosed | Strongest; the TMax recipe uses our settings | Weakest unless cross-checked | Good for Harbor work |

### Two findings that apply to every option

- **TRL's merged sync is probably wrong for the last chunk.** The adapter is un-merged before vLLM copies the final chunk of weights (layers 30–31 on Qwen3.5-4B by the researcher's arithmetic, which I did not recompute). The fix is a one-line clone.
- **A zero-learning-rate run cannot check weight sync** under any trainer, because the adapter delta is zero. A separate one-hour check with a non-zero adapter is needed before the pilot, which is the first run that learns.

### What to use vLLM for under any trainer

- **Sampling:** token-id prompts, explicit full-distribution settings, returned token ids and processed log-probabilities, one request per episode turn with a per-turn `max_tokens`.
- **Scoring existing sequences:** `prompt_logprobs=0` with `max_tokens=1`, for the sync check.
- **Pilot rollouts:** K reports from one frozen trajectory is one request with `n=K`.
- **Adapter hot-swap:** `/v1/load_lora_adapter` with a new adapter name per step, and only once it passes the sync check on Qwen3.5. Three vLLM bugs are open on that path.
- **Version:** keep 0.30.0 with TRL. verl needs its own pin, 0.29.0.

### Recommendation

1. **Run Stage 1 and the report-only pilot on stock TRL, one process.** The pilot is single-turn. The zero-learning-rate run tolerates serial tools (the review's estimate is $60–120, unmeasured) and measures what decides the trainer.
2. **Before the pilot,** apply the clone patch and run the sync check.
3. **For the full loop, default to TRL with a concurrent loop and switch to verl only on the triggers below.** `HonestGRPOTrainer` is not written yet, so nothing is sunk. At 64 episodes per step TRL trains two arms per node and verl's sync mode trains one, so throughput does not clearly favour verl (UNVERIFIED).
4. **Do not make C or D the primary path.** C costs as much as verl and is the hardest to defend. D needs the same custom work as verl and has an open correctness report at our scale.
5. **If calendar time matters more than engineering days,** start the GPU-free half of the verl port now and smoke-test it on one cheap GPU with a small model.

### Decision rule

**Measure in the zero-learning-rate run** (stock TRL, 20–30 rounds), per step:
- generation, grading, training and sync time;
- tool time, both the total and the slowest call per turn;
- peak trainer memory on the longest batch;
- tokens per episode, stop reasons and pass counts per group;
- the importance ratio.

Projected concurrent step time T = generation + the sum over turns of the slowest tool call + grading + training + sync.

**Stay on TRL if all hold:**
- T is at most 15 minutes;
- training is at most half of T;
- the longest batch fits at per-device batch size 1;
- at most 50% of groups have no reward contrast;
- at most 20% of episodes exhaust the token budget.

**Switch to verl if any holds:**
- the longest batch runs out of memory at batch size 1, or training is more than half of T (only verl trains on several GPUs);
- fixing contrast or exhaustion needs group size 16 or more, or a budget of 32k or more, which at least doubles tokens per step;
- after the pilot, per-token advantages or online alternative reports become part of the method.

**The sync check picks the sync mode, not the trainer.** Pass means a mean importance ratio within 1% of 1 and a mean log-probability gap no larger than the base model's, with no NaN, on 2k–20k-token sequences.
- **Merged with the clone patch passes:** keep it.
- **Only adapter-only passes:** use adapter sync. That is about a day in TRL (copy the versioned-adapter code from `AsyncGRPOTrainer`) and the default in verl.
- **Neither passes on Qwen3.5:** serve in float32 or stay on Qwen3-4B. verl does not help here.

The 1% and 50% thresholds come from the spec. The 15 minutes, one half and 20% are my judgement and should be fixed before the run. If a switch triggers, gate verl on its rollout-versus-actor log-probability metrics, run with `bypass_mode=False`.

---

Sources:
- Clones: `/private/tmp/claude-501/-Users-michael-Documents-Github-restraint/6df909e7-c51f-47be-83a7-a544cf10085e/scratchpad/trainer-options/` (`verl/verl` at 8718ca30, `vllm/vllm-v0.30.0`, `ecosystem/SkyRL` at 5f9a1be)
- TRL 1.14.1: `/Users/michael/Documents/Github/restraint/.venv/lib/python3.12/site-packages/trl/`
- Spec: `/Users/michael/Documents/Github/restraint/docs/SPEC_V3.md`