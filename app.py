"""Gradio frontend for the svdquant-kernels Phase 1b NPU launch smoke.

What this Space does:

1. On the FIRST button press, runs ``space/link_smoke.sh`` to link the
   pre-cross-built aarch64 .o files in ``space/objects/`` against the
   container's native aarch64 CANN runtime, producing
   ``space/svdquant_gemm_w4a4_smoke``.
2. Runs the resulting binary, which calls ``aclInit`` -> launches a
   placeholder ``gemm_w4a4`` kernel -> ``aclrtSynchronizeStream`` ->
   exits. Stdout/stderr are echoed back to the page.

The kernel itself is a no-op placeholder. This Space exists to verify
that the cross-build artifacts actually launch on a real 910B before
moving to Phase 2 (cube/vec coordination + real algorithm).
"""

from __future__ import annotations

import os
import shlex
import subprocess
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


def link_then_run() -> str:
    log: list[str] = []
    log.append(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<unset>')}")
    log.append(f"smoke binary present: {SMOKE_BIN.exists()}")
    if not SMOKE_BIN.exists():
        os.chmod(LINK_SCRIPT, 0o755)
        rc, out = _run(["bash", str(LINK_SCRIPT)])
        log.append(out)
        if rc != 0:
            return "\n".join(log) + "\n[FAIL] link step returned non-zero"
    rc, out = _run([str(SMOKE_BIN)])
    log.append(out)
    verdict = "[OK] kernel launched and stream synced" if rc == 0 else f"[FAIL] smoke exited {rc}"
    return "\n".join(log) + "\n" + verdict


with gr.Blocks(title="svdquant-kernels — Ascend 910B Phase 1b smoke") as demo:
    gr.Markdown(
        "# svdquant-kernels — Ascend 910B Phase 1b smoke\n"
        "Cross-built locally on x86_64 (CANN 8.5), final link runs on this 910B "
        "container. The kernel is a no-op placeholder — this only verifies the "
        "launch path itself."
    )
    btn = gr.Button("Run smoke")
    out = gr.Textbox(label="Output", lines=24, max_lines=60, show_copy_button=True)
    btn.click(fn=link_then_run, inputs=None, outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
