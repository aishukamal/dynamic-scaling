"""TUNIX-REWARD item 1: push the GOLD patch of calibrated R2E-Gym-Subset instances through the DeepSWE reward path exactly
as a training rollout calls it: SWEEnv(use_agent_sandbox) -> fleet.acquire (gVisor sandbox, agent-runtime + fresh eval
container) -> hide_r2e_tests_for_rollout -> [gold files written into /testbed of the agent-runtime container] ->
env.final_reward_fn() == the recipe's _compute_reward_in_fresh_container (extract_agent_patch via git diff, then
evaluate_patch_in_fresh_container: git apply + setup_env + r2egym _calculate_reward in the eval container).
Control per instance: a second, untouched sandbox graded the same way (what does the "default" patch look like?).
Output: JSON lines GOLDRESULT / CTRLRESULT on stdout (+ PATCHEXTRACT / PATCHDUMP / PATCHAPPLY lines from the patched
openhands_utils.py). Dataset order = the recipe's (load_from_disk + shuffle seed 42), rows ROW_IDX."""
import base64, json, logging, os, shlex, sys, threading, time, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)
log = logging.getLogger("gold")
sys.path.insert(0, "/app")
from examples.deepswe import sandbox_utils, swe_env, openhands_utils  # noqa: E402
from tunix.experimental.examples.deepswe_dist import deepswe  # noqa: E402

ROWS = [int(x) for x in os.getenv("ROW_IDX", "0,1,2,3").split(",")]
WITH_CONTROL = os.getenv("WITH_CONTROL", "1") == "1"
ds = deepswe.load_deepswe_dataset(dataset_name="R2E-Gym/R2E-Gym-Subset", dataset_split="train",
                                  dataset_path=os.getenv("DATASET_PATH", "/tmp/tunix-data/r2e-cal"),
                                  cache_dir=None, shuffle=True, seed=42)
log.info("dataset rows=%d columns=%s", len(ds), list(ds.column_names) if hasattr(ds, "column_names") else "?")
entries = []
for i in ROWS:
    e = deepswe._entry_at(ds, i)
    entries.append((i, e))
    pc = e.get("parsed_commit_content"); pc = json.loads(pc) if isinstance(pc, str) else pc
    log.info("ENTRY row=%d image=%s repo=%s commit=%s old_commit=%s n_file_diffs=%d keys=%s", i, e.get("docker_image"),
             e.get("repo_name"), e.get("commit_hash"), (pc or {}).get("old_commit_hash"), len((pc or {}).get("file_diffs", [])), sorted(e.keys()))

fleet = sandbox_utils.init_global_fleet(tasks=[e for _, e in entries], max_concurrency=8, num_generations=2,
                                        batch_size=len(entries), max_warmpool_replicas=2, scaffold="r2egym")
rewrite = sandbox_utils.get_image_rewrite_fn(getattr(fleet, "_image_rewrite_fn", None))
images = {}
for i, e in entries:
    img = str(e.get("docker_image")); images[i] = rewrite(img) if rewrite else img
log.info("fleet ready; warming %s", images)

def _warm(img):
    try:
        fleet.warm_image(img, replicas_override=2 if WITH_CONTROL else 1, wait=True)
        log.info("WARM done %s", img)
    except Exception as ex:  # pylint: disable=broad-exception-caught
        log.error("WARM failed %s: %s", img, ex)
ths = [threading.Thread(target=_warm, args=(img,)) for img in set(images.values())]
[t.start() for t in ths]; [t.join() for t in ths]

def gold_files(entry):
    pc = entry["parsed_commit_content"]; pc = json.loads(pc) if isinstance(pc, str) else pc
    out = {}
    for d in pc.get("file_diffs", []):
        plus = d.get("plus_file"); path = (plus if isinstance(plus, str) else (plus or {}).get("path", "")) or ""
        minus = d.get("minus_file"); mpath = (minus if isinstance(minus, str) else (minus or {}).get("path", "")) or ""
        path = path.removeprefix("b/"); mpath = mpath.removeprefix("a/")
        content = d.get("new_file_content")
        key = path if path and path != "/dev/null" else mpath
        out[key] = (content, mpath if mpath != "/dev/null" else None)
    return out

def exec_in(env, cmd, timeout=120):
    res = openhands_utils._exec_in_sandbox(env.env, cmd, timeout=timeout)
    return openhands_utils._unpack_exec_output(res)

def write_file(env, path, content):
    q = shlex.quote("/testbed/" + path)
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    tmp = "/tmp/gold_%d.b64" % (abs(hash(path)) % 100000)
    for idx in range(0, max(1, len(b64)), 49152):
        chunk = b64[idx:idx + 49152]; redir = ">" if idx == 0 else ">>"
        out, rc = exec_in(env, f"printf '%s' {shlex.quote(chunk)} {redir} {tmp}")
        if rc != 0: raise RuntimeError(f"chunk write failed rc={rc} {out[:200]}")
    out, rc = exec_in(env, f"mkdir -p $(dirname {q}) && base64 -d {tmp} > {q} && rm -f {tmp} && wc -c {q}")
    if rc != 0: raise RuntimeError(f"decode failed rc={rc} {out[:200]}")
    return out.strip()

def make_env(entry):
    env = swe_env.SWEEnv(entry=dict(entry), max_steps=20, step_timeout=1800, reward_timeout=1800, backend="kubernetes",
                         scaffold="r2egym", use_agent_sandbox=True, fleet=fleet)
    obs, info = env.reset()
    return env, obs, info

results = []
for i, e in entries:
    rec = {"row": i, "image": images[i], "repo": e.get("repo_name"), "commit": e.get("commit_hash")}
    # ---- gold ----
    try:
        t0 = time.time(); env, obs, info = make_env(e); t_acq = time.time() - t0
        head, _ = exec_in(env, "cd /testbed && git rev-parse HEAD && git status --porcelain | head -20 && ls /r2e_tests 2>&1 | head -3")
        rec["gold_sandbox_head_status"] = head[:600]
        files = gold_files(e); written = {}; skipped = []
        for path, (content, mpath) in files.items():
            if path.startswith("tests/") or "/tests/" in path or path.startswith("test/"):
                skipped.append(path); continue
            if content is None:
                out, rc = exec_in(env, f"rm -f {shlex.quote('/testbed/' + (mpath or path))}"); written[path] = f"deleted rc={rc}"
            else:
                written[path] = write_file(env, path, content)
        st, _ = exec_in(env, "cd /testbed && git status --porcelain | head -30 && git diff --stat | tail -3")
        rec.update({"gold_files_written": written, "gold_files_skipped_tests": skipped, "gold_git_status": st[:800], "acquire_s": round(t_acq, 1)})
        t1 = time.time(); r = env.final_reward_fn(); rec["gold_reward"] = float(r); rec["gold_reward_s"] = round(time.time() - t1, 1)
        log.info("GOLDRESULT %s", json.dumps(rec))
        env.close()
    except Exception as ex:  # pylint: disable=broad-exception-caught
        rec["gold_error"] = repr(ex) + " | " + traceback.format_exc()[-1500:]; log.error("GOLDRESULT %s", json.dumps(rec))
        try: env.close()
        except Exception: pass
    # ---- control: untouched sandbox through the same reward path ----
    if WITH_CONTROL:
        crec = {"row": i, "image": images[i]}
        try:
            env, obs, info = make_env(e)
            st, _ = exec_in(env, "cd /testbed && git rev-parse HEAD && git status --porcelain | head -20")
            crec["ctrl_git_status"] = st[:600]
            t1 = time.time(); r = env.final_reward_fn(); crec["ctrl_reward"] = float(r); crec["ctrl_reward_s"] = round(time.time() - t1, 1)
            log.info("CTRLRESULT %s", json.dumps(crec)); env.close()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            crec["ctrl_error"] = repr(ex) + " | " + traceback.format_exc()[-1500:]; log.error("CTRLRESULT %s", json.dumps(crec))
            try: env.close()
            except Exception: pass
    results.append(rec)
log.info("SUMMARY %s", json.dumps([{k: v for k, v in r.items() if k in ("row", "image", "gold_reward", "gold_reward_s", "gold_error")} for r in results]))
for img in set(images.values()):
    try: fleet.unwarm_image(img) if hasattr(fleet, "unwarm_image") else None
    except Exception as ex: log.warning("unwarm %s: %s", img, ex)
try:
    for name in ("close", "shutdown", "teardown"):
        if hasattr(fleet, name): getattr(fleet, name)(); log.info("fleet.%s done", name); break
except Exception as ex: log.warning("fleet close: %s", ex)
log.info("DONE")
