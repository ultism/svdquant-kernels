---
name: gitcode-space-iter
description: One-shot validation iteration on a GitCode (AtomGit) AI Community Space — push, trigger rebuild, wait for the smoke output, capture results, then pause the Space to release compute. Use when the user has just pushed a change to a Space's deployment branch and wants to verify it ran without manual page reloads. Requires the chrome-devtools MCP server to be configured and the Space's App + Setting pages to be open in the browser.
---

# GitCode Space iteration loop

Automate one validation cycle on a Space-hosted Gradio app:

1. Trigger a rebuild (auto-pulls latest commit from main)
2. Wait for the app's startup output via the container log panel
3. Capture the formatted result via the Gradio iframe
4. Pause the Space to free compute when done

This skill is **chrome MCP driven** — it operates the GitCode web UI rather than calling any GitCode API. That keeps it credential-free and works for any user already logged in to GitCode in the browser session that the chrome-devtools MCP is attached to.

## When to use

- After pushing a change to the `main` branch the Space deploys from
- When the app emits clear startup banners (e.g. `smoke OK` / `smoke FAILED`) that can serve as `wait_for` sentinels
- When you want to release the limited-time-free NPU/GPU resource right after the run

Skip this skill if:
- The user's app does not have a clear textual completion sentinel — the loop has no reliable exit signal
- The Space is on a different platform (HuggingFace Spaces, Modal, etc.) — selectors and lifecycle differ

## Required browser state

The user must have **two tabs** open in the chrome instance the MCP is attached to:

- App page: `https://ai.gitcode.com/<owner>/<repo>` — has the Gradio iframe + `日志` button
- Setting page: `https://ai.gitcode.com/<owner>/<repo>/setting` — has the `重置` and `暂停` buttons

`mcp__chrome-devtools__list_pages` confirms both are present before starting.

## The loop

For each step the **selector is the visible button text**, not a uid (uids change between sessions). Use `take_snapshot` on each page to find the current uid for a given button text.

### 1. Push the commit

Standard `git push` to whatever remote/branch the Space pulls from. No MCP involvement.

### 2. Click 重置 on the setting page

Selects the page, takes a snapshot, finds the button whose accessible name is `重置` under the `Space 管理` heading, clicks it. The button transitions to disabled — that is the local confirmation that the reset request landed.

`重置` triggers `git pull origin <default_branch>` + container rebuild + restart. There is no separate "redeploy" or "rebuild" button — `重置` is the canonical entry point for picking up new commits.

### 3. Reload the App page

`navigate_page type=reload` on the App tab to flush the previously running iframe. Without the reload, the old iframe stays cached and the next snapshot still shows the previous run's output.

### 4. Click 日志 on the App page

The container log panel is rendered into the host page DOM (same-origin), so `wait_for` can match against it. Without expanding this panel first, `wait_for` will only see the Space header chrome and timeout.

### 5. wait_for the startup sentinel

`wait_for` with the app's success and failure markers, e.g. `["smoke OK", "smoke FAILED"]`. Default timeout is short — set a longer one (60–120s) since rebuild + container start takes 30–90s on the free tier.

If the timeout fires, take a snapshot to see whether the page is still in `启动中` / `构建中` state and either retry the wait or report a stuck rebuild.

### 6. Capture the run output

**Success path:** `take_snapshot` on the App page — the Gradio iframe is cross-origin (served from `aihub-run.gitcode.com/online-space/<id>/`) but its accessibility tree IS exposed via Chrome DevTools Protocol. The Gradio Textbox's full `value` shows up in the snapshot output.

**Failure path:** the app likely called `sys.exit(non-zero)` so the iframe is gone. Use `evaluate_script` to read `document.body.innerText` instead — the log panel has the full traceback because it's the container's stdout/stderr.

### 7. Click 暂停 on the setting page

Switch back to the setting tab, snapshot, find the `暂停` button, click. Status badge transitions from `运行中` → `未启动`. `wait_for ["未启动"]` confirms the pause completed.

Skipping this step keeps the Space slot occupied and may exhaust the free quota faster.

## Important constraints

### Cross-origin iframe blocks `wait_for`

The Gradio app runs in an iframe served from a different domain (`aihub-run.gitcode.com`). `wait_for` walks `document.body.innerText` of the host page only — it cannot see iframe text. So:

- For startup detection, **search the log panel** (host page, same-origin) — that is why step 4 expands the panel before step 5
- For reading formatted output, **use `take_snapshot`** which reaches across origins via CDP

### Selectors are by visible text, not by uid

uids reset between sessions. Always re-snapshot before clicking; never reuse a uid from earlier turns. The button labels (`重置`, `暂停`, `日志`) are stable.

### Authentication URL gotcha

If pushing fails with a credential prompt (hangs), check whether the remote URL is `atomgit.com` versus `gitcode.com`. The two domains share repos but `~/.git-credentials` typically only stores a token for one. Switching the remote to `https://gitcode.com/<owner>/<repo>.git` lets git reuse the existing PAT.

### Resource limits

GitCode AI Community Spaces on the free tier (NPU 910B basic etc.) have queue / time limits. Pausing after each run avoids hitting them. The platform does not auto-pause.

## Adapting to a different app

The two app-side details to adjust:

- The success/failure sentinels — match what the deployed app prints to stdout at startup (any unique string works, ideally one that survives log truncation if the platform truncates)
- Whether to read the iframe textbox or just the log panel — depends on whether the app surfaces a structured result in its UI

Everything else (button labels, page navigation, lifecycle) is GitCode-specific and stays.
