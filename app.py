"""Gradio frontend for the svdquant-kernels NPU op extension test.

Phase 2f deployment flow on GitCode Space 910B:

  1. Container boot → app.py imports.
  2. Run `space/link_op_extension.sh` to compile + link
     `libop_extension.so` from pre-cross-built host_stub + kernel
     .o files plus the one host wrapper source. Links against the
     container's native torch + torch_npu (Space image carries them).
  3. `import op_extension` loads the .so, registering
     `torch.ops.svdquant.gemm_w4a4`.
  4. Launch pytest tests/ to drive the op end-to-end against
     baseline reference; capture stdout.
  5. Always launch the Gradio webui so the captured trace lives in
     a Textbox even when tests fail (otherwise GitCode's log panel
     blanks out once the container exits — see memory note
     `feedback_space_log_panel_blanks_on_exit.md`).

Phase 2e's smoke_main C++ flow (link_smoke.sh + standalone smoke ELF +
.bin argv shuffle) is fully retired by this script. Phase 3 keeps this
boot scaffolding unchanged; only the op signature in
`csrc/python/host/svdquant_w4a4_op.cpp` and the test in
`tests/test_gemm_w4a4.py` extend.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

# GitCode Space's container runs as uid=1000 but `/etc/passwd` has no
# matching entry, so `getpass.getuser()` fails with `KeyError: getpwuid()`
# the moment torch's `_inductor.codecache` initializes (it sanitizes the
# cache dir by username). Setting USER (or LOGNAME / LNAME / USERNAME)
# short-circuits getpass.getuser() before it falls back to pwd lookup.
# Subprocess calls (link script, pytest) inherit this env.
os.environ.setdefault("USER", "svdquant")
os.environ.setdefault("HOME", os.environ.get("HOME") or "/tmp")

import gradio as gr

ROOT = Path(__file__).resolve().parent
SPACE = ROOT / "space"
LINK_SCRIPT = SPACE / "link_op_extension.sh"
OP_EXT_SO = SPACE / "objects" / "libop_extension.so"


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as e:
        return 127, f"$ {pretty}\nFileNotFoundError: {e}\n"
    except subprocess.TimeoutExpired:
        return 124, f"$ {pretty}\n<<timed out after 300s>>\n"
    out = (
        f"$ {pretty}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\nexit: {proc.returncode}\n"
    )
    return proc.returncode, out


def _step(log: list[str], msg: str) -> None:
    """Stream-print so the container log keeps progress mid-flight.

    Necessary because GitCode's log panel blanks out as soon as the
    container exits — buffered output won't reach it. See
    `feedback_space_log_panel_blanks_on_exit.md`.
    """
    print(msg, flush=True)
    log.append(msg)


def link_then_test() -> tuple[bool, str]:
    import traceback
    log: list[str] = []
    _step(log, f"[boot] ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<unset>')}")
    _step(log, f"[boot] op_extension .so present: {OP_EXT_SO.exists()}")

    # Always re-link to pick up source/object changes; setuptools-style
    # mtime caching can come later. Link is ~5 s for one .cpp.
    _step(log, "[boot] linking libop_extension.so...")
    try:
        os.chmod(LINK_SCRIPT, 0o755)
    except OSError:
        pass
    rc, out = _run(["bash", str(LINK_SCRIPT)])
    _step(log, out)
    if rc != 0:
        _step(log, "[FAIL] link step returned non-zero")
        return False, "\n".join(log)

    _step(log, "[boot] running unittest tests/...")
    # PYTHONPATH adds csrc/python so `import op_extension` resolves.
    # We use stdlib `unittest` (not pytest) because Space's image
    # ships torch but not pytest and pip install may be blocked.
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{ROOT / 'csrc' / 'python'}:{ROOT}:" + env.get("PYTHONPATH", "")
    )
    # Run the test file directly via its `if __name__ == '__main__'`
    # entry rather than `unittest discover`, which would require
    # tests/ to be a Python package (an __init__.py we don't want
    # because it interferes with future pytest layouts). Direct
    # invocation is the safest cross-runner pattern.
    test_cmd = [sys.executable, str(ROOT / "tests" / "test_gemm_w4a4.py")]
    pretty = " ".join(shlex.quote(c) for c in test_cmd)
    try:
        proc = subprocess.run(
            test_cmd, env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        _step(log, f"$ {pretty}\n<<unittest timed out after 300s>>")
        return False, "\n".join(log)
    _step(log, f"$ {pretty}\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}\nexit: {proc.returncode}\n")
    if proc.returncode == 0:
        _step(log, "[OK] all tests passed")
        return True, "\n".join(log)

    _step(log, f"[FAIL] unittest exited {proc.returncode}")
    try:
        # Surface any uncaught exception in the test module.
        traceback.print_exc()
    except Exception:
        pass
    return False, "\n".join(log)


def _print_banner(title: str) -> None:
    bar = "=" * 72
    print(bar, flush=True)
    print(title, flush=True)
    print(bar, flush=True)


# === Run link + tests before launching the webui ===
_print_banner("svdquant-kernels op_extension — link + pytest")
_OK, INITIAL_OUTPUT = link_then_test()
_print_banner("tests OK" if _OK else "tests FAILED — webui still starts so trace is visible")


def rerun() -> str:
    _, txt = link_then_test()
    return txt


with gr.Blocks(title="svdquant-kernels — Ascend 910B torch.ops.svdquant.gemm_w4a4") as demo:
    gr.Markdown(
        "# svdquant-kernels — Ascend 910B `torch.ops.svdquant.gemm_w4a4`\n"
        "Phase 2f: link `libop_extension.so` from pre-cross-built device + "
        "host-launcher .o files plus the torch op wrapper source, register the op "
        "via `TORCH_LIBRARY_IMPL(svdquant, PrivateUse1, m)`, then run pytest. "
        "Boot output is captured below; if tests fail, this webui still "
        "starts so the trace stays reachable."
    )
    out = gr.Textbox(label="Output", lines=24, max_lines=80, value=INITIAL_OUTPUT)
    btn = gr.Button("Re-run link + tests")
    btn.click(fn=rerun, inputs=None, outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
