"""Gradio frontend for the svdquant-kernels NPU smoke.

Container-as-health-check semantics. On Space startup, before launching
the gradio webui, we link the pre-cross-built aarch64 .o files in
``space/objects/`` and run the resulting smoke binary. The full output
is dumped to stdout so the Space's container log keeps the trace
regardless of webui state.

If the smoke fails, the process exits non-zero — the Space sees a
crashed container and surfaces it; no webui ever starts. This matches
the user's "serverless-style" expectation where you read errors from
the log panel, not by clicking around in a UI that may not even render.

If the smoke succeeds, the gradio webui starts with the captured output
already shown in a Textbox, plus a Re-run button for retries.

The smoke binary self-identifies which phase it's testing (1b launch
path, 2a cube/vec dispatch, 2b cube fp16 mock GEMM, …); this app.py
is phase-agnostic and just propagates exit codes + stdout.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
SPACE = ROOT / "space"
LINK_SCRIPT = SPACE / "link_smoke.sh"
SMOKE_BIN = SPACE / "svdquant_gemm_w4a4_smoke"


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as e:
        return 127, f"$ {pretty}\nFileNotFoundError: {e}\n"
    except subprocess.TimeoutExpired:
        return 124, f"$ {pretty}\n<<timed out after 180s>>\n"
    out = f"$ {pretty}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\nexit: {proc.returncode}\n"
    return proc.returncode, out


def link_then_run() -> tuple[bool, str]:
    log: list[str] = []
    log.append(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<unset>')}")
    log.append(f"smoke binary present: {SMOKE_BIN.exists()}")
    if not SMOKE_BIN.exists():
        try:
            os.chmod(LINK_SCRIPT, 0o755)
        except OSError:
            pass
        rc, out = _run(["bash", str(LINK_SCRIPT)])
        log.append(out)
        if rc != 0:
            log.append("[FAIL] link step returned non-zero")
            return False, "\n".join(log)
    rc, out = _run([str(SMOKE_BIN)])
    log.append(out)
    if rc == 0:
        log.append("[OK] kernel launched and stream synced")
        return True, "\n".join(log)
    log.append(f"[FAIL] smoke exited {rc}")
    return False, "\n".join(log)


def _print_banner(title: str) -> None:
    bar = "=" * 72
    print(bar, flush=True)
    print(title, flush=True)
    print(bar, flush=True)


# === Container-as-health-check: run smoke before launching the webui ===
_print_banner("svdquant-kernels NPU smoke — running before webui")
ok, INITIAL_OUTPUT = link_then_run()
print(INITIAL_OUTPUT, flush=True)
_print_banner("smoke OK" if ok else "smoke FAILED — exiting before webui starts")

if not ok:
    # Crash the container so the Space log panel keeps the full trace
    # and the platform shows the run as failed. No webui to obscure it.
    sys.exit(1)


def rerun() -> str:
    _, txt = link_then_run()
    return txt


with gr.Blocks(title="svdquant-kernels — Ascend 910B NPU smoke") as demo:
    gr.Markdown(
        "# svdquant-kernels — Ascend 910B NPU smoke\n"
        "Cross-built locally on x86_64 (CANN 8.5), final link runs on this 910B "
        "container. The smoke binary itself reports which phase is being "
        "tested in its stdout; this UI is just a viewer.\n\n"
        "Smoke ran on container startup; output captured below. If it had "
        "failed, this webui would never have started — the Space's log panel "
        "would have the trace instead."
    )
    out = gr.Textbox(label="Output", lines=24, max_lines=60, value=INITIAL_OUTPUT)
    btn = gr.Button("Re-run")
    btn.click(fn=rerun, inputs=None, outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
