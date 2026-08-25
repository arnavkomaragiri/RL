# Agentic rollout logprob parity investigation

Date: 2026-08-22

## Acceptance criteria

The investigation treats each boundary separately instead of inferring input
correctness from a final KL metric.

| Boundary | Required check |
| --- | --- |
| Gym capture | Prompt IDs, generation IDs, and generation logprobs are unchanged. |
| Logical rollout | Every captured model call becomes exactly one independent segment. |
| MCore input | Segment lengths sum to the logical row length; no segment exceeds 131,072 tokens. |
| Loss | Every generated token has loss weight one exactly once; prompt tokens have weight zero. |
| Router replay | Installed routes match vLLM exactly; natural-route divergence is measured separately. |
| Output | No sequence is masked; token multiplicative error is below 1.05 and max sequence error is below 1.10. |

The output thresholds match the accuracy scale used by existing NeMo-RL
functional tests. They are not being used to relax any exact input invariant.

## Controls

| Run | Cache/routes | Result |
| --- | --- | --- |
| Qwen3 30B-A3B, 16x1, one step | R3 enabled; all 428 calls carried routes | Completed in 28 minutes. Zero sequences masked. Sequence multiplicative error 1.0158-1.0224 (mean 1.0190), token error 1.0182, generation KL 0.0008. |
| Nemotron 3.5, 2x2, one step | FP32 vLLM SSM cache; no R3 | Zero sequences masked. Sequence error 1.1652-1.6894 (mean 1.3318), token error 1.2875, generation KL 0.0118. |
| Nemotron 3.5, 16x1, two steps | BF16 vLLM SSM cache; no R3 | Step 1 masked 16/16 sequences. Step 2 masked 14/16. Step-1 sequence error ranged from 2.424 to 5,220,971. |

The Qwen control uses a different architecture and R3, so it does not by
itself distinguish Mamba state precision from route replay. It does falsify a
shared Gym capture, independent-call packing, or generic MCore logprob failure
as the cause of the catastrophic Nemotron result.

## 1. Captured token sequences

`examples/nemo_gym/audit_token_pipeline.py` imports Gym's production prefix
builder and independent-call projection. It then reconstructs the NeMo-RL row
and matches diagnostic records by total and trainable lengths. For every
reported outlier it verifies the token ID, captured generation logprob, loss
offset, attention segment, and context window.

Results:

- Nemotron BF16 control: 97 captures, 5,744 calls, 5,744 training segments,
  32/32 diagnostic records uniquely matched, zero invariant errors.
- Nemotron FP32 control: 8 captures, 438 calls, 438 training segments, 4/4
  diagnostic records uniquely matched, zero invariant errors.
- Qwen control: 32 captures, 428 calls, 428 training segments, zero invariant
  errors. All 428 calls carried routed-expert metadata.
- No audited call had unresolved retries, quarantined calls, empty
  generations, or a segment above 131,072 tokens.
- The BF16 control's 32 train rows contain 1,860 independent calls and
  1,650,191 generated tokens. Every diagnostic generation logprob matched the
  Gym capture at `1e-6` tolerance.

Conclusion: token capture and projection are exact. Retokenization is not the
source of the observed model mismatch.

## 2. Attention and loss layout

`_use_exact_nemo_gym_call_sequences` concatenates exact call tensors only as a
logical transport row and records `packed_attention_segment_lengths`.
`expand_batched_data_for_packed_attention` expands that row back into ordinary
causal call sequences before behavior-logprob recomputation, reference
logprobs, and training. Results are reassembled only after each forward.

For the audited BF16 batch, MCore therefore processed 1,860 independent causal
sequences rather than 32 multi-million-token sequences. The audit proves:

- the segment lengths reproduce the captured call boundaries exactly;
- no attention edge can cross a call boundary;
- each segment has prompt zeros followed by generation ones in its loss mask;
- the generation-token total equals the trainable-token total; and
- every diagnostic loss offset maps to the corresponding captured generation
  offset.

Conclusion: prefix-tree flattening is not changing model context, and no token
is optimized twice.

## 3. Model internals

### Mamba state precision

Nemotron 3.5 contains 23 Mamba, 23 MoE, and 6 attention layers. Its HF config
declares `mamba_ssm_cache_dtype: float32`.

The bundled MCore `mamba_ssm` implementation calls `_chunk_state_fwd` with
`states_in_fp32=True`. Its state-passing Triton kernel loads and accumulates
state in FP32 and allocates the final state as FP32. The installed version does
not expose the newer explicit MCore state-dtype argument, but the recurrence is
still internally FP32.

vLLM allocates the temporal cache from `mamba_ssm_cache_dtype`, passes its
actual dtype to prefill, stores the returned state in that cache, and reads and
updates it on every decode. A BF16 setting therefore quantizes persistent state
at every transition, while FP32 matches the checkpoint and MCore recurrence.

The A/B result is consistent with accumulated recurrent-state error:

- BF16 retained outliers have a positive correlation with call index
  (`r=0.278`) and generation-relative position (`r=0.151`).
- Their retained mean absolute error rises from 6.03 nats for calls 0-4 to
  8.13 for calls 20-49 and 11.59 for calls 100+.
- FP32 retained outliers have no call-index trend (`r=-0.025`); call buckets
  remain near 4.0-4.6 nats.

These correlations use only the retained top-32 errors per selected sequence;
the next diagnostic records aggregate statistics over every trainable token.

Conclusion: BF16 SSM cache is a demonstrated major cause, but FP32 is not
sufficient for the required tolerance.

### MoE routing

Qwen replayed vLLM routes and remained within tolerance. The existing
Nemotron controls did not request routes, so MCore selected experts from its
own slightly different hidden states. Freezing router weights does not freeze
the top-k decisions.

The R3 diagnostic now records both:

1. whether the route used by MCore exactly matches the installed vLLM route;
2. the route MCore would naturally select from the same scores, including
   exact ordered matches, expert-set matches, and mean top-k overlap.

This separates route transport/replay correctness from route sensitivity.

### Weight refit

The refit stream has a manifest that rejects duplicate, unexpected, missing,
or failed IPC weights. The Nemotron vLLM loader also raises on unknown model
parameters and returns the loaded parameter set, although NeMo-RL currently
does not consume that return value. Refit remains a residual hypothesis, but
the strong cache A/B and successful Qwen dummy-load/refit path make a total
refit failure unlikely. If FP32 plus exact route replay remains outside
tolerance, parameter coverage/checksums are the next model-boundary check.

## 4. Output error structure

The FP32 control retained 128 token outliers:

- absolute difference 2.659-10.066 nats (median 3.862);
- 125 positive signed differences and 3 negative;
- no decode/re-encode mismatches or non-finite values;
- errors occur from the first generated token through the final decile; and
- error magnitude has negligible correlation with prompt length (`r=0.033`)
  or call index (`r=-0.025`).

The BF16 control retained 1,024 token outliers:

- absolute difference 1.531-26.426 nats (median 7.121);
- 1,015 positive signed differences and 9 negative;
- no non-finite values; and
- increasing error with later calls and later decode position.

The positive sign is expected under behavior-policy sampling:
`E[token~P_gen](log P_gen - log P_train) = KL(P_gen || P_train) >= 0`.
It is evidence of distribution divergence, not independently evidence of a
sign or token-alignment bug.

## Active falsification experiment

Slurm job `16373697` failed during Megatron actor construction before generation:
its inherited `mamba_training_ssm_states_dtype: float32` setting is unsupported by
the container's installed `mamba_ssm`. It produced no route or logprob evidence.

Replacement job `16375214` completed all 16 selected rollouts and MCore
behavior-logprob inference from snapshot
`harbor-opencode-nemotron35-parity-r3-defaultstate-20260822-194536`. No sequence
crossed the existing `2.0` multiplicative-error masking threshold. Training then
failed during activation-checkpoint backward with `IndexError: pop from empty
list` in MCore router replay.

The router trace identified 24 replay consumers for a model with 23 base-policy
MoE layers. The extra consumer is the auxiliary MTP block: its local layer
numbering restarts at two, so it was incorrectly assigned the base policy's
layer-two rollout routes. Its natural route comparison had zero exact expert-set
matches in the sampled microbatch, confirming that it is not the corresponding
policy layer. Rollout payloads do not carry MTP routes, so the corrected replay
enumeration excludes every router under an `mtp` subtree.

The natural-route diagnostic was also the only added top-k computation attached
to the training graph. It now receives `scores.detach()` while retaining the
caller's grad mode. A surrounding `torch.no_grad()` is not equivalent because
this MCore version uses `torch.is_grad_enabled()` to request sorted training
top-k output. The rerun therefore tests both the replay-consumer cardinality fix
and removal of the diagnostic branch from autograd without changing its top-k
semantics.

Corrected job `16379455` was submitted from immutable snapshot
`harbor-opencode-nemotron35-parity-r3-mtpfix-20260822-142836`. It retains the
same 16x1 one-step workload and all other model, cache, replay, and diagnostic
settings from job `16375214`.

Job `16379455` completed successfully, including activation-checkpoint
backward and vLLM weight refit. Across both the previous-logprob and training
forwards, the trace covered all 23 base-policy MoE layers with zero replay
mismatches and zero missing-route fallbacks. The replay queue remained
nonempty before and after every traced forward; the erroneous MTP replay
consumer was absent.

MCore's natural routes were materially different from vLLM's captured routes:

- 64.36% of token rows had the same ordered top-k experts;
- 86.22% had the same unordered expert set; and
- mean expert overlap was 97.60%.

Forcing the captured routes reduced the Nemotron parity error from the FP32
no-replay control's 1.3318 mean sequence multiplicative error and 0.0118
generation KL to:

- 1.0067-1.0159 sequence multiplicative error (mean 1.0096);
- 1.0100 token multiplicative error;
- 0.00104 generation KL; and
- 0.000268 Jensen-Shannon divergence.

No sequence was masked. All 16 selected rollouts were fully rebuilt from model
calls, with a delivered-token fraction of 1.0, no failed rollout rows, no
sandbox cleanup failures, no honeypot accesses, and one accepted terminal judge
score per rollout. The slowest rollout was not stuck: policy execution used
55m48s, both deadline alerts were delivered and processed, and verification
used 5m27s.

The run does have one orthogonal training-objective caveat. It used G=1, so all
GRPO advantages and the reported policy loss were zero, but the recipe did not
set `mtp_loss_scaling_factor`. MCore therefore used its default factor of 0.1;
the run logged MTP loss 0.1799, MTP acceptance 90.71%, and total gradient norm
0.6492. This does not affect the pre-update parity comparison, but the optimizer
step was driven by the auxiliary MTP objective rather than GRPO. Production RL
must explicitly decide whether to retain that objective; set the factor to zero
for a pure policy-gradient update.

Interpretation:

- If errors fall to the Qwen range and natural routes frequently differ, MoE
  route selection explains the FP32 residual.
- If replay is exact but errors remain near 1.33, route divergence is not
  sufficient; compare prefix/chunk scheduling next, then refit parameter
  coverage and same-input intermediate states.
- If replayed routes do not match installed routes, stop: that is an R3 data
  alignment bug and output comparisons are not interpretable.
- If input or segment invariants fail, stop before attributing anything to the
  model.

The next experiments should change one factor at a time: route replay, then
prefix/chunk scheduling, then refit versus direct HF load. A BF16 cache should
not be used for further parity experiments because it contradicts the model
config and has already failed decisively.
