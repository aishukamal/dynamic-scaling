# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""verl AgentLoop that drives a SWE agent against a GKE agent-sandbox.

Per trajectory: claim a sandbox for the instance image, loop
generate -> parse bash block -> exec in sandbox -> append observation tokens
(mask 0), until the model submits or a budget trips. On submit, extract the
non-test diff and grade it in a FRESH sandbox (pristine grading, see
pristine_grader.py); the reward rides back on AgentLoopOutput.reward_score.

Token accounting follows verl's ToolAgentLoop: LLM tokens are appended
verbatim (never re-tokenized), observations are tokenized as delta user
messages with remove_system_prompt=True.

Register via agent_loop_config_path (see examples/swe_agent_loop.yaml);
dataset rows select it with agent_name == "swe_agent". Requires
data.return_raw_chat=True and the RBAC in configs/swe_sandbox_rbac.yaml.

Compatible with the verl build on the cluster (0.9.0.dev interface:
server_manager.generate returns TokenOutput); tolerates older builds where
generate returned a bare token id list.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer  # type: ignore[import-not-found]

from integration.verl.swe.pristine_grader import grade_patch_pristine
from integration.verl.swe.sandbox import SandboxClient, get_thread_client, make_name

# TPU-M3 loop v3.1: all sandbox I/O (create + wait_ready, exec, grade, delete) runs on a dedicated thread pool instead of the
# event loop's default executor (min(32, cpu + 4) threads per process). With 100+ live trajectories per worker process the
# default pool filled with blocking sandbox calls and the per-turn tokenizer.decode calls behind them stopped, so no new
# generate request left the process (tpum3-dynr2: LB in-flight 0 with 350 sandboxes Running). Concurrency towards the API
# server is still capped by the semaphores in sandbox.py; the pool only decouples decode from sandbox I/O.
_SANDBOX_IO_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("SWE_IO_THREADS", "256")), thread_name_prefix="swe-sandbox-io")
from integration.verl.swe.scaffold import (
    NO_COMMAND_OBSERVATION,
    build_system_prompt,
    format_observation,
    is_submit,
    parse_turn,
    extra_blocks_note,
    is_comment_only,
    SUBMIT_REJECTED_OBSERVATION,
    REPEATED_COMMAND_OBSERVATION,
    COMMENT_ONLY_OBSERVATION,
)

logger = logging.getLogger(__name__)

DIFF_CMD = "cd /testbed && git add -N . >/dev/null 2>&1; git diff HEAD"
MAX_DIFF_BYTES = 512 * 1024

# R2E images ship with a dirty git state (the harness's baked-in edits).
# Committing it as a baseline right after sandbox start makes `git diff HEAD`
# agent-only — the fresh grading sandbox has the identical dirty working tree,
# so an agent-only patch applies cleanly there.
BASELINE_CMD = (
    "cd /testbed && git add -A >/dev/null 2>&1; "
    "git -c user.email=swe@llm-d.dev -c user.name=swe commit -qm baseline >/dev/null 2>&1; true"
)


def default_system_prompt_ids(tokenizer: Any) -> list[int]:  # noqa: ANN401
    """Token ids of the system prompt the chat template injects for a conversation
    without an explicit system message (verl <= 0.9 ``initialize_system_prompt``).
    Used to strip that prefix when tokenizing delta user messages."""
    ids = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
    return list(ids["input_ids"] if isinstance(ids, Mapping) else ids)


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        rollout = self.rollout_config
        self.response_length: int = rollout.response_length
        multi_turn = rollout.get("multi_turn", None)
        self.max_assistant_turns: int = (
            multi_turn.get("max_assistant_turns") if multi_turn else None
        ) or 32
        self._init_tunables()
        self._init_chat_template_compat()

    # verl 0.10.0.dev removed AgentLoopBase.apply_chat_template (replaced by the
    # continuous-token ct_* API) and verl.utils.chat_template. Keep the validated
    # 0.9 tokenization path for this loop (rl-pool-autoscaler PoC compat shim;
    # token discipline unchanged: LLM tokens verbatim, observations as delta user
    # messages with the default system prompt stripped).
    def _init_chat_template_compat(self) -> None:
        if not hasattr(self, "system_prompt"):
            self.system_prompt = default_system_prompt_ids(self.tokenizer)

    if not hasattr(AgentLoopBase, "apply_chat_template"):

        async def apply_chat_template(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            *,
            remove_system_prompt: bool = False,
            **kwargs: Any,  # noqa: ANN401
        ) -> list[int]:
            def _tokenize() -> list[int]:
                ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **(getattr(self, "apply_chat_template_kwargs", None) or {}),
                )
                ids = list(ids["input_ids"] if isinstance(ids, Mapping) else ids)
                return ids[len(self.system_prompt):] if remove_system_prompt else ids

            return await self.loop.run_in_executor(None, _tokenize)

    def _init_tunables(self) -> None:
        """Initialize env-tunable knobs in one place.

        Grouped so test drivers that bypass __init__ can initialize them in
        one call (see integration_check.py).
        """
        self.namespace = os.getenv("SWE_SANDBOX_NAMESPACE", "agents-system")
        self.cmd_timeout_s = int(os.getenv("SWE_CMD_TIMEOUT_S", "60"))
        self.obs_max_chars = int(os.getenv("SWE_OBS_MAX_CHARS", "6000"))
        # Cover autoscaler scale-up + node-cold pulls of multi-GiB task images.
        self.sandbox_wait_s = int(os.getenv("SWE_SANDBOX_WAIT_S", "600"))
        # Rollout sandboxes idle between commands; small requests pack more per
        # node (CPU quota is the concurrency ceiling). Bursts use the limit,
        # not the request. Grading sandboxes keep beefier defaults - graded
        # test runs are timing-sensitive.
        self.sandbox_cpu_request = os.getenv("SWE_SANDBOX_CPU_REQUEST", "500m")
        self.sandbox_mem_request = os.getenv("SWE_SANDBOX_MEM_REQUEST", "1Gi")
        # TPU-M3 reward work: reject a submit while `git diff HEAD` is empty (the one-shot-plan mode ended 35 % of the
        # trajectories on turn 1 with no edit), and dump every turn's raw text/command/rc to a JSONL (per worker file).
        self.submit_guard = os.getenv("SWE_SUBMIT_GUARD", "1") == "1"
        self.turn_dump = os.getenv("SWE_TURN_DUMP", "")
        # A command identical to the previous turn's is not re-run (the 1.7B model loops on one command: 62 % of consecutive
        # turns identical in tpum3-rwa); comment-only blocks are not run either. Both answer with an instruction.
        self.repeat_guard = os.getenv("SWE_REPEAT_GUARD", "1") == "1"

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:  # noqa: C901,PLR0912,PLR0914,PLR0915,ANN401
        extra_info = dict(kwargs["extra_info"])
        messages = [dict(m) for m in kwargs["raw_prompt"]]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_prompt(messages[0]["content"])
        prompt_ids: list[int] = await self.apply_chat_template(messages)

        full_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        has_logprobs = True
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        sandbox_name = make_name("swe", extra_info["instance_id"])
        assistant_turns = user_turns = 0
        submitted = False
        reward, reason = 0.0, "no-submit"
        # verl >= 0.10 stamps the model-weight version on every TokenOutput
        # (extra_fields global_steps / min_global_steps / max_global_steps) and
        # the fully-async trainer requires them on the final output (partial-
        # rollout staleness accounting). Mirror ToolAgentLoop: keep the first
        # turn's stamps, bump max_global_steps on later turns.
        gen_extra: dict[str, Any] = {}
        stats: dict[str, int] = {"turns": 0, "wellformed": 0, "multi_block": 0, "executed": 0, "rc0": 0, "submit_rejected": 0,
                                 "patch_nonempty": 0, "repeated": 0, "comment_only": 0}
        prev_command: str | None = None
        # TPU-M3 observability (local overlay only): per-turn trace for 1 in 20 trajectories (command as parsed, rc, observation head).
        trace = (int(request_id[:4], 16) % 20) == 0
        trace_turns: list[str] = []

        client = None
        # Boot the sandbox concurrently with the first generate: turn 1 never
        # needs the sandbox until its command executes, and sandbox startup
        # (scheduling + image pull) is tens of seconds.
        sandbox_future = self.loop.run_in_executor(
            _SANDBOX_IO_POOL, lambda: self._create_sandbox(sandbox_name, extra_info["docker_image"], stats)
        )
        try:
            while (
                assistant_turns < self.max_assistant_turns
                and len(response_mask) < self.response_length
            ):
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=full_ids,
                        sampling_params=sampling_params,
                    )
                token_ids = getattr(output, "token_ids", output)
                log_probs = getattr(output, "log_probs", None)
                out_extra = getattr(output, "extra_fields", None) or {}
                if not gen_extra:
                    gen_extra.update(out_extra)
                else:
                    for key in ("global_steps", "max_global_steps"):
                        if out_extra.get(key) is not None:
                            gen_extra[key] = out_extra[key]
                budget = self.response_length - len(response_mask)
                token_ids = list(token_ids)[:budget]
                full_ids += token_ids
                response_mask += [1] * len(token_ids)
                if log_probs is None:
                    has_logprobs = False
                elif has_logprobs:
                    response_logprobs += list(log_probs)[:budget]
                assistant_turns += 1
                if len(response_mask) >= self.response_length:
                    reason = "token-budget"
                    break

                text = await self.loop.run_in_executor(
                    None, lambda t=tuple(token_ids): self.tokenizer.decode(t)
                )
                command, n_blocks = parse_turn(text)
                stats["turns"] += 1
                stats["wellformed"] += 1 if command is not None else 0
                stats["multi_block"] += 1 if n_blocks > 1 else 0
                if trace:
                    trace_turns.append(f"T{assistant_turns} blocks={n_blocks} think={'<think>' in text} cmd={(command or 'NO-BASH-BLOCK')[:120]!r}")
                rc: int | None = None
                if command is not None and is_submit(command):
                    if assistant_turns == 1:
                        # TPU-M3 observability (local overlay only): why does a trajectory submit on its first turn?
                        logger.info("swe_agent %s: TURN1-SUBMIT text_head=%r text_tail=%r",
                                    extra_info.get("instance_id"), text[:400], text[-200:])
                    rejected = False
                    if self.submit_guard:
                        with simple_timer("tool_calls", metrics):
                            if client is None:
                                client = await sandbox_future
                            rc_d, diff_now = await self.loop.run_in_executor(
                                _SANDBOX_IO_POOL, lambda: client.exec(sandbox_name, DIFF_CMD, timeout=120)
                            )
                        rejected = rc_d == 0 and not diff_now.strip()
                    if not rejected:
                        submitted = True
                        self._dump_turn(extra_info, request_id, assistant_turns, text, command, n_blocks, None, "submit")
                        break
                    stats["submit_rejected"] += 1
                    observation = SUBMIT_REJECTED_OBSERVATION
                elif command is None:
                    observation = NO_COMMAND_OBSERVATION
                elif is_comment_only(command):
                    stats["comment_only"] += 1
                    observation = COMMENT_ONLY_OBSERVATION
                elif self.repeat_guard and prev_command is not None and command.strip() == prev_command.strip():
                    stats["repeated"] += 1
                    observation = REPEATED_COMMAND_OBSERVATION
                else:
                    with simple_timer("tool_calls", metrics):
                        if client is None:
                            client = await sandbox_future
                        rc, out = await self.loop.run_in_executor(
                            _SANDBOX_IO_POOL, lambda c=command: self._run_command(client, sandbox_name, c)  # noqa: B023
                        )
                    stats["executed"] += 1
                    stats["rc0"] += 1 if rc == 0 else 0
                    observation = format_observation(rc, out, self.obs_max_chars)
                    if n_blocks > 1:
                        observation += extra_blocks_note(n_blocks, "last" if os.getenv("SWE_BLOCK_POLICY", "first") == "last" else "first")
                prev_command = command
                self._dump_turn(extra_info, request_id, assistant_turns, text, command, n_blocks, rc, observation)
                if trace:
                    trace_turns.append(f"  -> rc={rc if rc is not None else 'n/a'} obs={observation[:160]!r}")

                obs_ids = await self.apply_chat_template(
                    [{"role": "user", "content": observation}], remove_system_prompt=True
                )
                if len(response_mask) + len(obs_ids) >= self.response_length:
                    reason = "token-budget"
                    break
                full_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                if has_logprobs:
                    response_logprobs += [0.0] * len(obs_ids)
                user_turns += 1

            if assistant_turns >= self.max_assistant_turns and not submitted:
                reason = "max-turns"

            if submitted:
                with simple_timer("tool_calls", metrics):
                    if client is None:
                        client = await sandbox_future
                    reward, reason = await self.loop.run_in_executor(
                        _SANDBOX_IO_POOL, lambda: self._grade(client, sandbox_name, extra_info, stats)
                    )
        except Exception as e:  # sandbox/infra failure: salvage the trajectory  # noqa: BLE001
            if not gen_extra:
                # TPU-M3 fix (local overlay): no generate output was ever received (e.g. the serving replica died before the
                # first turn), so the output would carry no weight-version stamps and the V1 trainer fails on int(None) when
                # tagging it (tpum3-dyn step 13). Re-raise: verl marks the prompt group failed and refills it.
                logger.warning("swe_agent rollout error before any generation on %s: %s (re-raised for refill)", extra_info.get("instance_id"), e)
                raise
            logger.warning("swe_agent rollout error on %s: %s: %s", extra_info.get("instance_id"), type(e).__name__,
                           str(e).replace("\n", " | ")[:300])
            reward, reason = 0.0, f"rollout-error: {type(e).__name__}"
        finally:
            if client is None:
                # Consume the boot future's result/exception so it never goes unobserved.
                try:  # noqa: SIM105
                    client = await sandbox_future
                except Exception:  # noqa: BLE001,S110
                    pass
            # Delete the Sandbox CR unconditionally: create may have succeeded
            # even when wait_ready timed out (pod stuck Pending).
            try:  # noqa: SIM105
                await self.loop.run_in_executor(
                    _SANDBOX_IO_POOL, lambda: get_thread_client(self.namespace).delete(sandbox_name)
                )
            except Exception:  # noqa: BLE001,S110
                pass

        response_ids = full_ids[len(prompt_ids):]
        if not response_ids:
            # verl's padding path can't handle empty responses ('list' has no
            # attribute 'dim'); emit one loss-masked pad token instead.
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            response_ids = [pad_id]
            response_mask = [0]
            response_logprobs = [0.0]
        # TPU-M3 observability (local overlay only): per-trajectory generate / tool seconds for the gen-vs-tool split.
        if trace:
            logger.info("swe_agent %s: TRACE submitted=%s reason=%s reward=%s turns=%d\n%s",
                        extra_info.get("instance_id"), submitted, reason, reward, assistant_turns, "\n".join(trace_turns)[:6000])
        logger.info(
            "swe_agent %s: reward=%s reason=%s turns=%d resp_tokens=%d gen_s=%.1f tool_s=%.1f stats=%s",
            extra_info.get("instance_id"), reward, reason, assistant_turns, len(response_ids),
            float(metrics.get("generate_sequences", 0.0)), float(metrics.get("tool_calls", 0.0)), json.dumps(stats),
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if has_logprobs else None,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            reward_score=reward,
            extra_fields={**gen_extra, "swe_reason": reason, "swe_submitted": submitted, **{f"swe_{k}": v for k, v in stats.items()}},
        )

    def _dump_turn(self, extra_info: dict, request_id: str, turn: int, text: str, command: str | None,
                   n_blocks: int, rc: int | None, observation: str) -> None:
        """Per-turn JSONL record (SWE_TURN_DUMP=<path prefix>; one file per worker process): raw assistant text, parsed
        command, block count, return code and observation head. Append-only, best effort."""
        if not self.turn_dump:
            return
        try:
            path = f"{self.turn_dump}.{os.getpid()}.jsonl"
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "instance_id": extra_info.get("instance_id"),
                   "request_id": request_id, "turn": turn, "text": text, "command": command, "n_blocks": n_blocks,
                   "rc": rc, "observation_head": observation[:400], "think": "<think>" in text,
                   "think_closed": "</think>" in text}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug("turn dump failed: %s", e)

    # --- sync helpers, always called via run_in_executor -------------------

    def _create_sandbox(self, name: str, image: str, stats: dict | None = None):
        client = get_thread_client(self.namespace)
        t0 = time.monotonic()
        client.create(
            name, image,
            cpu_request=self.sandbox_cpu_request,
            memory_request=self.sandbox_mem_request,
        )
        t1 = time.monotonic()
        waited = client.wait_ready(name, timeout=self.sandbox_wait_s)
        t2 = time.monotonic()
        client.exec(name, BASELINE_CMD, timeout=120)
        if stats is not None:
            # loop v3.2 diagnostics (per trajectory, printed in the completion line): create call, client-observed wait_ready,
            # server-side pod create-to-ready, readiness polls and time spent waiting for the control-plane semaphore.
            stats["sbx_create_s"] = round(t1 - t0, 1)
            stats["sbx_wait_s"] = round(waited, 1)
            stats["sbx_pod_ready_s"] = None if client.last_pod_ready_s is None else round(client.last_pod_ready_s, 1)
            stats["sbx_polls"] = client.last_wait_polls
            stats["sbx_sem_s"] = round(client.last_wait_sem_s, 1)
            stats["sbx_baseline_s"] = round(time.monotonic() - t2, 1)
        return client

    def _run_command(self, client: SandboxClient, name: str, command: str) -> tuple[int, str]:
        import shlex

        wrapped = f"cd /testbed && timeout {self.cmd_timeout_s} sh -c {shlex.quote(command)}"
        return client.exec(name, wrapped, timeout=self.cmd_timeout_s + 30)

    def _grade(self, client, name: str, extra_info: dict, stats: dict | None = None) -> tuple[float, str]:
        rc, diff = client.exec(name, DIFF_CMD, timeout=120)
        if stats is not None and diff.strip():
            stats["patch_nonempty"] = 1
        if rc != 0:
            return 0.0, f"diff-failed: {diff[-200:]}"
        if len(diff) > MAX_DIFF_BYTES:
            return 0.0, "diff-too-large"
        return grade_patch_pristine(extra_info, diff, namespace=self.namespace)
