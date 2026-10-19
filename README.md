# dynamic-scaling

Code for the elastic sampler pool PoC: dynamic sampler scaling for agentic RL on TPU. A small controller runs outside the RL framework, reads the job's own metrics (sampler slot utilisation, trainer wait, queue depth), and scales sampler hosts in and out while training continues. Two stacks were built: Tunix/JAX (JobSet based rollout workers, Raiden weight sync) and verl on TorchTPU (Ray based standalone sampler pool, TPU hardware plugin). Both drive SWE agent rollouts against gVisor sandboxes on GKE.

## Layout

    controller/          shared rl_pool_autoscaler package: policy, Tunix backend, Kubernetes and RayCluster actuators, tests
    tunix/patch/         final Tunix patch set shipped by ConfigMap: metrics exporter, patched deepswe entrypoint, native tool calling, parser
    tunix/reward/        reward measurement patches, DeepSWE arm scripts, gold grader
    tunix/k8s/           render, apply, teardown, kill and analysis scripts, controller ConfigMap builder, RBAC, example prep pod
    tunix/runs/          rendered manifests and render.env of the pristine Tunix runs
    tunix/image/         Cloud Build and Dockerfiles for the base, raiden, deepswe and clean Tunix images
    verl/plugin/         TPU hardware plugin tree at commit 3a4b8e1 (fork of verl-hardware-plugin) with the pool actor and pool control
    verl/plugin-patches/ pool-*.patch series, the final applied one is pool-3a4b8e1.patch
    verl/loop/           SWE agent loop overlay v3.2 (swe_work) and swe_loop_v3.2.diff against the loop source
    verl/image/          Dockerfile.m3, cloudbuild.yaml, import checks, baseline swe_work as baked into the image
    verl/controller/     controller launcher, pool control client, polling and head pod prep
    verl/ops/            run, kill, ladder, cleanup and analysis scripts, runtime envs of the pristine arms
    verl/k8s/            RayCluster, RBAC, staging, probe and sandbox smoke manifests
    sandbox/             TPU capacity expansion script and the sandbox image mirror Cloud Build
    tools/figures/       timeline, chip duty and progress figure renderers plus the HTML builder

## Pristine runs

Tunix (tunix/runs):

- rc2b: dynamic, 24 steps, newest sampler killed after step 13
- rs4c, rs4d: fixed 8:4 (trainer chips : sampler chips)
- rs8b: fixed 8:8
- rs12b: fixed 8:12
- rl60: long dynamic, 60 steps
- rl60b: resume of rl60 at lr 1e-5, steps 60 to 89

verl (verl/ops runtime envs):

- dynr4: dynamic, 24 versions, sampler killed at version 13
- s84c3, s88c2, s812c: fixed 8:4, 8:8, 8:12

## Not in this repo

Logs, trajectory dumps, metrics CSV and JSONL files, figures, datasets, model files and the written reports are intentionally left out. Image references point at a private Artifact Registry and node names refer to a cluster that no longer exists in this form, so the manifests are a record of the configuration rather than something to apply as is.
