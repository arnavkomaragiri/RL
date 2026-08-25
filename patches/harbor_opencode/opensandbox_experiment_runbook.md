# OpenSandbox experiment runbook

## 2026-08-22 task-root data exposure

The 32x16 run mounted each Harbor task root at `/data` instead of mounting only
its capsule payload at `environment/data`. The policy therefore saw
`task.toml`, `steps/`, and hidden verifier inputs through `/app/data -> /data`,
while the actual scientific files appeared one level deeper at
`/app/data/environment/data`.

Rows 272 and 370 both spent most of their policy budget diagnosing this layout,
then replaced `/app/data` with copied capsule files. The separate verifier
correctly rejected the changed compatibility-link topology, but the redundant
missing judge-strace check masked that verifier-start failure. The launcher now
appends `environment/data` to per-task S3 roots by default, the judge no longer
requires a filesystem trace, and shared-workspace validation failures are
logged before Harbor records the row result.

This contaminated the full run: 1,075 of 1,087 captured policy trajectories
accessed task-root metadata, and 635 explicitly referenced `task.json` or
`judge_instruction.md` from the hidden tests tree. Rewards and policy-hacking
rates from that run are not valid experimental results.

## 2026-08-20 volume-template incident

A stress run dispatched 2,048 sandbox creates with a task-data volume whose
host path ended in the literal placeholder `{task_name}`. Every pod therefore
requested a path that did not exist and could not start. The queue was flooded
before the first create failure could stop further dispatch.

The local artifacts contain 2,088 Harbor trial configurations with the literal
placeholder across 136 task names. Of those, 1,024 recorded the 1,800-second
environment-start timeout, with those timeout files arriving within roughly 71
seconds of one another. New trial configurations continued to appear after the
first wave timed out. The cluster-side count of 2,048 pods therefore exceeded
the logical 1,024-environment concurrency bound.

### Root cause

- The launch configuration used `{task_name}`.
- The Harbor sandbox renderer only documented and expanded
  `{environment_name}`, `{task_id}`, and `{session_id}`.
- Unknown placeholders were silently passed to the provider.
- The scale run did not gate fan-out on a successful sandbox using the exact
  task-specific volume configuration.
- Environment timeouts released rollout slots while physical sandbox creates
  were still present, and provider create retries could issue another create
  for the same logical rollout. A logical concurrency bound alone did not bound
  cluster objects during failure.

The affected volume was the task-data S3 host mount, not the EFS workspace
itself. Both use the same provider-options template path, so the guard applies
to all sandbox volumes, sandbox-local path copies, and compatibility symlinks.

### Corrective changes

- `{task_name}` is accepted as an alias for `{environment_name}`.
- Unsupported placeholder names fail Gym configuration before rollout
  dispatch and fail again before `provider.create()` as defense in depth.
- The Slurm launcher validates volume, path-copy, and symlink JSON and
  placeholder names before submitting a job.
- Direct policy workspaces use the physical source
  `<EFS host>/<owner>/nemo-gym-harbor-artifacts/<run>/{context_id}`. The policy
  mounts it read-write at `/app`; the verifier mounts the same source read-only
  at `/app`. Both roles independently mount only the clean capsule payload at
  `<task>/environment/data` from S3 read-only at `/data`; the Harbor task root,
  instructions, and hidden verifier files are not exposed through that mount.
  The policy creates `/app/data -> /data`, while the verifier only validates the
  inherited link. Only a marker and digest pass through Harbor.
- Context cleanup mounts only the run parent in an owner-labeled maintenance
  sandbox and removes the exact context-ID child after verifier termination.
- Regression tests cover task-name rendering and prove an unknown placeholder
  cannot reach the sandbox provider.

### Direct workspace canary

The production OpenSandbox endpoint was checked with the pinned BixBench image
and one real train-task S3 source. A context-specific EFS host path that did not
previously exist was mounted read-write at `/app`, while task data was mounted
read-only at `/app/data`. A fresh verifier then mounted the same EFS source
read-only at `/app`: it read the policy report and the task data, and a write
attempt failed with `Read-only file system`. This initial canary established
read-only handoff semantics but used nested mounts; the production layout now
uses sibling `/app` and `/data` mounts plus `/app/data -> /data` to avoid
provider mount-order races. A final owner-labeled sandbox mounted only the run
parent and removed the exact context directory. All three sandboxes were
destroyed after the check.

## Scale-up checklist

1. Pin the image by digest and preserve the resolved launch configuration in
   the run snapshot.
2. Select one real task from the exact split and render every provider option.
   Reject any remaining `{identifier}` token.
3. Create one sandbox with the production image, entrypoint, metadata, resource
   requests, and rendered S3/EFS volumes. Verify `/data` contains only the
   expected capsule files (not `task.toml`, `steps/`, or the task's hidden
   verifier context) and `/app/data` resolves exactly to `/data`.
4. Run one complete policy and verifier episode through the production path.
   A generic image-only probe is insufficient for volume changes.
5. Scale in stages: 1 environment, one small prompt group, then the intended
   queue size. Do not dispatch the next stage until sandbox creation and task
   setup succeed at the current stage.
6. Watch create-failure cardinality. Repeated identical failures should halt
   admission rather than consume the remaining queue.
7. Cleanup must query by the exact ownership label
   `nemo-gym.nvidia.com/user=<owner>`, verify every returned object's metadata
   client-side, and only then terminate those IDs. Never use an unfiltered list
   from a cluster-wide API key as a deletion set.

## Follow-up guardrails

The template checks prevent this specific failure from reaching OpenSandbox.
Two broader protections remain useful for large experiments:

- Add a configuration-fingerprint circuit breaker that stops new creates after
  a small number of identical environment-start failures.
- Treat physical create attempts as capacity until termination is confirmed;
  do not release admission solely because the Harbor environment-start timer
  expired.
- Make the single-task production-volume canary an automated prerequisite for
  launchers that request high sandbox concurrency.
