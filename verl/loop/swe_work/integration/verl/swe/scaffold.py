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
"""Pure helpers for the SWE agent scaffold (no verl / kubernetes imports).

Protocol (mini-swe-agent style): each assistant turn contains reasoning plus
exactly one fenced bash block. The block is executed in the sandbox and its
output returned as the next user message. A block containing only ``submit``
ends the episode and triggers grading.
"""

from __future__ import annotations

import os
import re

SUBMIT_COMMAND = "submit"

BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)  # legacy (TPU-M3 ladder/dyn runs); see parse_turn

# Fence line: ``` plus an optional language tag, optionally followed by content on the same line (``` ``bash ls`` `` is seen
# from small models). Scanning line by line keeps opening and closing fences apart, so prose between two blocks is never
# taken for a command (the legacy regex matched a closing fence as an opening one whenever an opening fence was not in
# its expected form, executing the prose: rc 127 "This: not found" in 80 of 608 executed commands, TPU-M3 section 3.2).
FENCE_LINE_RE = re.compile(r"^[ \t]*```([A-Za-z0-9_+.-]*)[ \t]*(.*?)[ \t]*$")
BASH_TAGS = frozenset({"", "bash", "sh", "shell", "console", "zsh"})
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BLOCK_POLICY = os.getenv("SWE_BLOCK_POLICY", "first")  # which fenced block of a turn is executed: first | last

# Paths the agent is not allowed to influence at grading time. Chunks of the
# agent's patch touching these are stripped before pristine-sandbox grading.
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$|_test\.py$|(^|/)conftest\.py$|(^|/)r2e_tests(/|$)"
)

NO_COMMAND_OBSERVATION = (
    "No bash code block found in your reply. Respond with your reasoning and "
    "exactly one fenced bash block, e.g.:\n```bash\nls\n```\n"
    "Do not repeat a command you already ran. "
    f"When you are completely done, submit with:\n```bash\n{SUBMIT_COMMAND}\n```"
)

SUBMIT_REJECTED_OBSERVATION = (
    "submit rejected: `git diff HEAD` is empty, you have not changed any file yet. "
    "Inspect the code (grep, sed -n), make the fix with shell commands (sed -i or a heredoc), "
    "verify it, then submit."
)


REPEATED_COMMAND_OBSERVATION = (
    "You already ran exactly this command in your previous turn; its output is above and has not changed. "
    "Do something different: list the directory (ls), open another file (sed -n '1,80p' path), edit with sed -i or a "
    "heredoc, or run the tests."
)

COMMENT_ONLY_OBSERVATION = (
    "Your bash block contained only comments or was empty, so nothing ran. Give one real shell command in the block."
)


def is_comment_only(command: str) -> bool:
    """True when every line of the block is blank or a # comment (the model 'narrating' inside the fence)."""
    return all((not ln.strip()) or ln.lstrip().startswith("#") for ln in command.splitlines())


def extra_blocks_note(n_blocks: int, executed: str) -> str:
    return (
        f"\nNote: your reply contained {n_blocks} fenced blocks; only the {executed} one was executed. "
        "Send ONE command per turn and wait for its output before the next step."
    )


def strip_thinking(text: str) -> str:
    """Drop Qwen3-style <think>...</think> reasoning; an unterminated <think> (budget ran out while thinking) leaves nothing."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return THINK_RE.sub("", text)


def parse_bash_blocks(text: str) -> list[str]:
    """Every fenced block of the (thinking-stripped) text whose tag is a shell tag, in order; prose is never a block."""
    blocks: list[str] = []
    open_tag: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = FENCE_LINE_RE.match(line)
        if m and open_tag is None:
            open_tag, inline = m.group(1).lower(), m.group(2)
            buf = [inline] if inline else []
            continue
        if m and open_tag is not None:  # any fence line closes the open block (trailing text after ``` is dropped)
            if open_tag in BASH_TAGS:
                blocks.append("\n".join(buf).strip())
            open_tag, buf = None, []
            continue
        if open_tag is not None:
            buf.append(line)
    return [b for b in blocks if b]


def parse_turn(text: str, policy: str | None = None) -> tuple[str | None, int]:
    """(command, number of shell blocks) for one assistant turn: the FIRST shell block by default (a one-shot plan then
    runs step by step instead of ending on its final ``submit``), the last with SWE_BLOCK_POLICY=last."""
    blocks = parse_bash_blocks(strip_thinking(text))
    if not blocks:
        return None, 0
    pol = policy or BLOCK_POLICY
    return (blocks[-1] if pol == "last" else blocks[0]), len(blocks)


def parse_bash_command(text: str) -> str | None:
    """Extract the command of a turn (see parse_turn), or None."""
    return parse_turn(text)[0]


def is_submit(command: str) -> bool:
    return command.strip().lower() == SUBMIT_COMMAND


def truncate_output(text: str, max_chars: int) -> str:
    """Middle-truncate command output: keep head and tail, mark the elision."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return f"{text[:keep]}\n...[{len(text) - 2 * keep} chars truncated]...\n{text[-keep:]}"


def format_observation(returncode: int, output: str, max_chars: int) -> str:
    body = truncate_output(output, max_chars) if output else "(no output)"
    return f"<returncode>{returncode}</returncode>\n<output>\n{body}\n</output>"


def _patch_paths(chunk: str) -> list[str]:
    """Both sides of a `diff --git a/... b/...` chunk header, repo-relative."""
    m = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
    return [m.group(1), m.group(2)] if m else []


def filter_patch(diff_text: str, path_re: re.Pattern = TEST_PATH_RE) -> str:
    """Drop per-file chunks whose path matches path_re (test files).

    The remainder is what gets applied and graded in the pristine sandbox -
    a policy that edits tests gains nothing from it.
    """
    if not diff_text.strip():
        return ""
    kept = []
    for chunk in re.split(r"(?m)^(?=diff --git )", diff_text):
        if not chunk.strip():
            continue
        paths = _patch_paths(chunk)
        if paths and any(path_re.search(p) for p in paths):
            continue
        kept.append(chunk)
    return "".join(kept)


def build_system_prompt(base_prompt: str) -> str:
    """Append the tool protocol to the dataset's task-framing system prompt."""
    return (
        f"{base_prompt}\n\n"
        "## Interaction protocol\n"
        "You are in a live shell session at the root of the repository (the "
        "current directory; start with `ls` to see its layout). In every "
        "reply, briefly reason about the next step, then give EXACTLY ONE "
        "fenced bash block with one command to run, e.g.:\n"
        "```bash\ngrep -rn \"pattern\" . --include=*.py | head -20\n```\n"
        "The command's output will be returned to you before you write the next "
        "step; do not plan several commands in one reply. Each command runs in a "
        "fresh shell from the repository root: cd and environment variables do "
        "NOT persist between turns. Edit files with shell commands (heredocs, "
        "sed). Long outputs are truncated - prefer targeted commands (head, "
        "grep). Only after you have edited the code and verified the fix, end the "
        f"session with:\n```bash\n{SUBMIT_COMMAND}\n```\n"
        "Submitting without any file change is rejected."
    )
