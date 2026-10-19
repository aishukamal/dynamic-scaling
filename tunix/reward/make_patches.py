#!/usr/bin/env python3
"""TUNIX-REWARD measurement patches (console logging only; no algorithm change).
Produces patch/<basename>.py for ConfigMap tunix-reward-patch:
  run_gsm8k_dist_grpo.py, run_deepswe_dist.py, sandbox_utils.py : copies of the prior agents' observability patches
  swe_agent.py            : RAWRESP (raw model response + parsed function name per turn) and RAWOBS (env observation head)
  openhands_utils.py      : PATCHEXTRACT (git diff exit/bytes) and PATCHDUMP (patch bytes, sha1, files, first 2000 chars)
  inprocess_vllm_sampler_adapter.py : GENCALL/GENOUT for the first 3 generate calls per worker (sampling params, rendered prompt head/tail, output text/tokens)
Each anchor must match exactly once."""
import pathlib, shutil, subprocess
SRC = pathlib.Path.home() / "workspaces/tunix-poc"
OPS = pathlib.Path.home() / "workspaces/rl-pool-autoscaler-ops"
OUT = pathlib.Path(__file__).resolve().parent / "patch"
OUT.mkdir(exist_ok=True)

def patch(rel, edits, base=None):
    src = base or (SRC / rel)
    s = pathlib.Path(src).read_text()
    for old, new in edits:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    dst = OUT / pathlib.Path(rel).name
    dst.write_text(s)
    diff = subprocess.run(["diff", "-u", str(SRC / rel), str(dst)], capture_output=True, text=True).stdout
    (OUT / (pathlib.Path(rel).name + ".patch")).write_text(diff)
    print(rel, "->", dst.name, f"(+{diff.count(chr(10)+'+')-1} lines)")

# 0) prior agents' patched files, unchanged
for src, name in [(OPS/"tunix/patch/run_gsm8k_dist_grpo.py", "run_gsm8k_dist_grpo.py"),
                  (OPS/"tunix-deepswe/patch/run_deepswe_dist.py", "run_deepswe_dist.py"),
                  (OPS/"tunix-deepswe/patch/sandbox_utils.py", "sandbox_utils.py")]:
    shutil.copy(src, OUT / name); print("copied", name)

# 1) swe_agent.py
patch("examples/deepswe/swe_agent.py", [
    ("""    # Update Trajectory
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str
    cur_step.model_response = response
""",
     """    # Update Trajectory
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str
    cur_step.model_response = response
    try:  # TUNIX-REWARD measurement log
      logging.info(
          "RAWRESP agent=%x step=%d fn=%r resp_len=%d resp=%r",
          id(self), self.step, getattr(action, "function_name", ""),
          len(response or ""), (response or "")[:1500],
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
"""),
    ("""    observation = str(observation)
    if info is None:
      info = {}
    # If it's the first step in environment, let's apply user prompt template
""",
     """    observation = str(observation)
    if info is None:
      info = {}
    try:  # TUNIX-REWARD measurement log
      logging.info(
          "RAWOBS agent=%x step=%d done=%s obs_len=%d obs=%r",
          id(self), self.step, done, len(observation), observation[:700],
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    # If it's the first step in environment, let's apply user prompt template
"""),
])

# 2) openhands_utils.py
patch("examples/deepswe/openhands_utils.py", [
    ("""    raw_patch, exit_code = _unpack_exec_output(res)
    if exit_code != 0:
      logging.warning(
          "[SWEEnv] Failed to extract git patch from rollout container (exit=%s)",
""",
     """    raw_patch, exit_code = _unpack_exec_output(res)
    logging.info(  # TUNIX-REWARD measurement log
        "PATCHEXTRACT exit=%s raw_bytes=%d commit=%s",
        exit_code, len(raw_patch or ""), commit_ref,
    )
    if exit_code != 0:
      logging.warning(
          "[SWEEnv] Failed to extract git patch from rollout container (exit=%s)",
"""),
    ("""  if not patch or not patch.strip():
    logging.info("[SWEEnv] Empty git patch extracted; returning reward 0.0.")
    return 0.0
""",
     """  try:  # TUNIX-REWARD measurement log
    import hashlib as _hl  # pylint: disable=g-import-not-at-top
    _p = patch or ""
    logging.info(
        "PATCHDUMP bytes=%d sha1=%s files=%r head=%r",
        len(_p), _hl.sha1(_p.encode("utf-8", "replace")).hexdigest()[:12],
        re.findall(r"^diff --git a/(\\S+)", _p, flags=re.M)[:20], _p[:2000],
    )
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  if not patch or not patch.strip():
    logging.info("[SWEEnv] Empty git patch extracted; returning reward 0.0.")
    return 0.0
"""),
    ("""    apply_res = runtime.run(apply_cmd, timeout=60)
    apply_output, apply_exit = _unpack_exec_output(apply_res)
""",
     """    apply_res = runtime.run(apply_cmd, timeout=60)
    apply_output, apply_exit = _unpack_exec_output(apply_res)
    logging.info(  # TUNIX-REWARD measurement log
        "PATCHAPPLY exit=%s out=%r", apply_exit, (apply_output or "")[:500]
    )
"""),
])

# 3) inprocess_vllm_sampler_adapter.py
patch("tunix/experimental/rollout/inprocess_vllm_sampler_adapter.py", [
    ("""    if has_token_prompts:
      sampler_call_kwargs["input_strings"] = None
      sampler_call_kwargs["prompt_token_ids"] = prompt_token_ids_batch
    else:
      sampler_call_kwargs["input_strings"] = prompts
""",
     """    if has_token_prompts:
      sampler_call_kwargs["input_strings"] = None
      sampler_call_kwargs["prompt_token_ids"] = prompt_token_ids_batch
    else:
      sampler_call_kwargs["input_strings"] = prompts
    try:  # TUNIX-REWARD measurement log (first 3 generate calls per worker)
      self._gencall_n = getattr(self, "_gencall_n", 0) + 1
      if self._gencall_n <= 3:
        _p0 = prompts[0] if prompts else None
        _t0 = prompt_token_ids_batch[0] if prompt_token_ids_batch else None
        logging.info(
            "GENCALL n=%d reqs=%d token_prompts=%s temperature=%r top_p=%r"
            " top_k=%r max_gen=%r logprobs=%r prompt0_len=%s prompt0_head=%r"
            " prompt0_tail=%r token_prompt0_len=%s token_prompt0_tail=%r",
            self._gencall_n, len(requests), has_token_prompts, temperature,
            top_p, top_k, max_generation_steps, return_logprobs,
            (len(_p0) if isinstance(_p0, str) else None),
            (_p0[:400] if isinstance(_p0, str) else None),
            (_p0[-400:] if isinstance(_p0, str) else None),
            (len(_t0) if _t0 is not None else None),
            (list(_t0[-12:]) if _t0 is not None else None),
        )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
"""),
    ("""    prompt_lengths = getattr(sampler_output, "prompt_lengths", None)
    responses = []
    for i, req in enumerate(requests):
""",
     """    try:  # TUNIX-REWARD measurement log
      if getattr(self, "_gencall_n", 0) <= 3:
        _t = sampler_output.text
        _tx0 = _t[0] if isinstance(_t, list) and _t else _t
        _tk = sampler_output.tokens
        _tk0 = _tk[0] if isinstance(_tk, list) and _tk else None
        logging.info(
            "GENOUT n=%d text0_len=%s ntok0=%s last_tokens=%r text0=%r",
            self._gencall_n,
            (len(_tx0) if isinstance(_tx0, str) else None),
            (len(_tk0) if _tk0 is not None else None),
            (list(_tk0[-6:]) if _tk0 is not None else None),
            (_tx0[:700] if isinstance(_tx0, str) else None),
        )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    prompt_lengths = getattr(sampler_output, "prompt_lengths", None)
    responses = []
    for i, req in enumerate(requests):
"""),
])
print("done")
