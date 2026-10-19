"""Native Qwen3 tool calling for the r2egym scaffold (TUNIX-REWARD2). stdlib only.

TOOLCALL_MODE=qwen_json renders the four r2egym tools the way Qwen3's chat template renders `tools=` (a "# Tools"
block with JSON function signatures and the <tool_call>{json}</tool_call> instruction) and parses the model's
<tool_call> JSON back into the scaffold's Action (function_name + string parameters). Aliases map the model's own
conventions (top-level view/create/str_replace, bash/run, submit, file_path/content/old_string) onto the real tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

REAL_TOOLS = ("file_editor", "execute_bash", "search", "finish")

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "file_editor",
        "description": ("Custom editing tool for viewing, creating and editing files. State is persistent across calls. "
                        "view: if path is a file, shows `cat -n` output (optionally a line range); if a directory, lists it up to 2 levels. "
                        "create: creates a new file with file_text (fails if the file exists). "
                        "str_replace: replaces old_str (must match exactly one or more consecutive lines, unique in the file, without line numbers) with new_str. "
                        "insert: inserts new_str after line insert_line. undo_edit: reverts the last edit to path."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"], "description": "The command to run."},
            "path": {"type": "string", "description": "Absolute path to file or directory, e.g. /testbed/file.py or /testbed."},
            "file_text": {"type": "string", "description": "Required for create: the content of the file to be created."},
            "old_str": {"type": "string", "description": "Required for str_replace: the exact string in path to replace."},
            "new_str": {"type": "string", "description": "Replacement string for str_replace; required string to insert for insert."},
            "insert_line": {"type": "integer", "description": "Required for insert: new_str is inserted after this line number."},
            "view_range": {"type": "array", "items": {"type": "integer"}, "description": "Optional for view on a file: [start_line, end_line]; [start_line, -1] to the end of file."},
            "concise": {"type": "boolean", "description": "Optional for view; defaults to true (skeletal view). Set false for the full content."}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "execute_bash",
        "description": ("Execute a bash command in the terminal. Long-running commands should be backgrounded with output redirected. "
                        "Exit code -1 means the process is still running: call again with an empty cmd to get more logs, send input via cmd, or cmd=\"ctrl+c\" to interrupt."),
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string", "description": "The bash command (and optional arguments) to execute."}},
            "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": ("Search for a term in a directory or a single file. Directory (default .): recursive search of non-hidden files, truncated above 100 matching files. "
                        "File: grep -n with line numbers."),
        "parameters": {"type": "object", "properties": {
            "search_term": {"type": "string", "description": "The term or string to search for in files."},
            "path": {"type": "string", "description": "The file or directory to search in. Defaults to . if not specified."}},
            "required": ["search_term"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Finish the interaction once the task is complete or if no further progress can be made. The submit command finalizes your output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["submit"], "description": "Currently allowed value: submit."},
            "result": {"type": "string", "description": "The result text or final message to submit. Defaults to an empty string."}},
            "required": ["command"]}}},
]


def tools_block(one_call_note: bool = True) -> str:
    """The text Qwen3's chat template appends to the system message when `tools=` is given."""
    lines = ["# Tools", "", "You may call one or more functions to assist with the user query.", "",
             "You are provided with function signatures within <tools></tools> XML tags:", "<tools>"]
    lines += [json.dumps(t, ensure_ascii=False) for t in TOOL_SCHEMAS]
    lines += ["</tools>", "",
              "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
              "<tool_call>", '{"name": <function-name>, "arguments": <args-json-object>}', "</tool_call>"]
    if one_call_note:
        lines += ["", "Call exactly one function per response, after a short reasoning in natural text. Only the four functions above exist; "
                  "view, create, str_replace, insert and undo_edit are commands of file_editor, not functions."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- parsing
_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)

_EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}
_NAME_ALIASES = {
    "file_editor": "file_editor", "str_replace_editor": "file_editor", "editor": "file_editor", "edit_file": "file_editor",
    "read_file": "file_editor", "write_file": "file_editor", "create_file": "file_editor",
    "execute_bash": "execute_bash", "bash": "execute_bash", "execute": "execute_bash", "run": "execute_bash", "shell": "execute_bash",
    "execute_command": "execute_bash", "run_command": "execute_bash", "terminal": "execute_bash", "run_bash": "execute_bash",
    "search": "search", "grep": "search", "search_files": "search", "find": "search",
    "finish": "finish", "submit": "finish", "done": "finish", "complete": "finish", "final_answer": "finish",
}
_PARAM_ALIASES = {
    "file_editor": {"file_path": "path", "filepath": "path", "filename": "path", "file": "path", "content": "file_text", "text": "file_text",
                    "contents": "file_text", "old_string": "old_str", "new_string": "new_str", "old": "old_str", "new": "new_str",
                    "line": "insert_line", "line_number": "insert_line", "range": "view_range", "cmd": "command", "action": "command"},
    "execute_bash": {"command": "cmd", "commands": "cmd", "script": "cmd", "bash": "cmd", "code": "cmd"},
    "search": {"query": "search_term", "term": "search_term", "pattern": "search_term", "text": "search_term", "keyword": "search_term",
               "directory": "path", "dir": "path", "file": "path"},
    "finish": {"message": "result", "answer": "result", "summary": "result", "final": "result", "text": "result"},
}
_READ_LIKE = {"read_file": "view", "write_file": "create", "create_file": "create"}


def _first_json_object(s: str) -> dict | None:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s).rstrip("`").strip()
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:  # noqa: BLE001
        pass
    # repairs seen live (rwA): a trailing ')' where '}' was meant ('...[1, 10]})'), or an unclosed outer object
    base = s.rstrip(")\n ")
    for cand in (base + "}", base + "}}", base.rstrip("}") + "}}", s.replace("})", "}}", 1)):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            continue
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(s[i:j + 1])
                    return v if isinstance(v, dict) else None
                except Exception:  # noqa: BLE001
                    return None
    return None


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False)


def normalize(name: str, args: dict | None) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Map (model name, arguments) -> (real tool, string params, info). info['raw_valid'] = the model's own name was
    one of the four real tools; info['mapped'] = an alias mapping was applied; unknown names return ('', {}, info)."""
    args = dict(args or {})
    raw = (name or "").strip()
    low = raw.lower()
    info: dict[str, Any] = {"raw_name": raw, "raw_valid": raw in REAL_TOOLS, "mapped": False}
    if low in _EDITOR_COMMANDS:
        tool = "file_editor"
        args.setdefault("command", low)
        info["mapped"] = True
    elif low in _NAME_ALIASES:
        tool = _NAME_ALIASES[low]
        if low in _READ_LIKE:
            args.setdefault("command", _READ_LIKE[low])
        info["mapped"] = tool != raw
    else:
        info["reason"] = "unknown_function"
        return "", {}, info
    params: dict[str, str] = {}
    aliases = _PARAM_ALIASES.get(tool, {})
    for k, v in args.items():
        kk = aliases.get(str(k), str(k))
        if kk not in params or not params[kk]:
            params[kk] = _stringify(v)
    if tool == "finish":
        params.setdefault("command", "submit")
        if params.get("command") != "submit":
            params["command"] = "submit"
    if tool == "file_editor" and "command" in params:
        params["command"] = params["command"].strip().lower()
        if params["command"] in ("execute_bash", "bash", "run", "execute") and (params.get("cmd") or params.get("path")):
            # the model put another tool's name into file_editor's command field (seen live: 11 of 1704 turns)
            cmd = params.get("cmd") or params.get("command_line") or params.get("path")
            info["mapped"] = True
            return "execute_bash", {"cmd": cmd}, info
        if params["command"] in ("search", "grep") and (params.get("search_term") or params.get("path")):
            info["mapped"] = True
            return "search", {k: v for k, v in (("search_term", params.get("search_term") or params.get("path")), ("path", params.get("path") if params.get("search_term") else None)) if v}, info
        if params["command"] not in _EDITOR_COMMANDS:
            info["reason"] = "bad_editor_command"
    if tool == "execute_bash" and "cmd" not in params:
        info["reason"] = "missing_cmd"
    if tool == "search" and "search_term" not in params:
        info["reason"] = "missing_search_term"
    if tool == "file_editor" and "path" not in params and params.get("command") != "undo_edit":
        info["reason"] = "missing_path"
    return tool, params, info


def parse_qwen_json(response_text: str) -> tuple[str, str, dict[str, str], dict[str, Any]]:
    """Returns (thought, tool, params, info) for the FIRST <tool_call> JSON block; tool == '' when none parses."""
    text = response_text or ""
    m = _TC_RE.search(text)
    if not m:
        return text, "", {}, {"format": "none"}
    thought = text[: m.start()]
    obj = _first_json_object(m.group(1))
    if obj is None:
        return thought, "", {}, {"format": "tool_call_bad_json", "body": m.group(1)[:200]}
    name = obj.get("name") or obj.get("function") or obj.get("tool") or ""
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        obj = name
        name = obj.get("name", "")
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters") or obj.get("args") or {k: v for k, v in obj.items() if k not in ("name", "function", "tool")}
    if isinstance(args, str):
        parsed = _first_json_object(args)
        args = parsed if parsed is not None else {"cmd": args}
    tool, params, info = normalize(str(name), args if isinstance(args, dict) else {})
    info["format"] = "tool_call_json"
    return thought, tool, params, info
