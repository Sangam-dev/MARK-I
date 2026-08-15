import { app, BrowserWindow, ipcMain, screen } from "electron";
import { fileURLToPath } from "url";
import path from "path";
import { spawn } from "child_process";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
if (process.platform === "linux") {
    app.commandLine.appendSwitch("ozone-platform-hint", "x11");
    app.commandLine.appendSwitch("enable-transparent-visuals");
}
// ── Backend process lifecycle ────────────────────────────────────────────────
// Launching AURA also launches its own KANCHA backend (FastAPI + WebSocket,
// see kancha/api/server.py) so there's no separate terminal to remember to
// start. See answers/guide.md "Electron process lifecycle" for the full
// picture, including the PyInstaller-packaging follow-up for end users who
// don't have `uv`/Python installed.
//
// electron-out/main.js (this file, compiled) lives at
// jarvis_frontend/electron-out/ — the Python backend root is two levels up.
const BACKEND_DIR = path.join(__dirname, "..", "..");
const BACKEND_PORT = process.env.AURA_BACKEND_PORT || "8765";
// Set AURA_SKIP_BACKEND_SPAWN=1 during development if you're already running
// `uv run uvicorn api.server:app --reload` yourself in a separate terminal.
const SKIP_BACKEND_SPAWN = process.env.AURA_SKIP_BACKEND_SPAWN === "1";
let backendProcess = null;
function startBackend() {
    if (SKIP_BACKEND_SPAWN) {
        console.log("[AURA] AURA_SKIP_BACKEND_SPAWN=1 — not spawning backend, expecting it to already be running");
        return;
    }
    console.log(`[AURA] starting backend: uv run python scripts/run_backend.py --port ${BACKEND_PORT}  (cwd=${BACKEND_DIR})`);
    // `detached: true` makes the backend its own process group (Linux), so a
    // `process.kill(-pid)` in stopBackend() can kill the WHOLE tree — the `uv`
    // wrapper, the python launcher/uvicorn, and any grandchild — instead of
    // just the `uv` process, whose python uvicorn child used to survive,
    // orphaned, and hold the port until the next launch hit EADDRINUSE.
    backendProcess = spawn("uv", [
        "run",
        "python",
        "scripts/run_backend.py",
        "--host",
        "127.0.0.1",
        "--port",
        BACKEND_PORT,
    ], { cwd: BACKEND_DIR, stdio: "inherit", detached: process.platform !== "win32" });
    backendProcess.on("error", (err) => {
        // Most common cause: `uv` is not on PATH. The renderer's WebSocket client
        // (src/app/lib/wsClient.ts) will keep retrying regardless, so the UI
        // stays usable (with panels falling back to mock data) until this is
        // fixed and AURA is relaunched.
        console.error("[AURA] failed to start backend — is `uv` installed and on PATH?", err);
    });
    backendProcess.on("exit", (code, signal) => {
        console.log(`[AURA] backend process exited (code=${code}, signal=${signal})`);
        backendProcess = null;
    });
}
function stopBackend() {
    if (!backendProcess || backendProcess.killed)
        return;
    console.log("[AURA] stopping backend");
    if (process.platform !== "win32" && backendProcess.pid) {
        // Negative pid = the whole process group (see startBackend's detached
        // spawn). This is what actually reaches the python uvicorn.
        try {
            process.kill(-backendProcess.pid, "SIGTERM");
            return;
        }
        catch (err) {
            // Fall through to killing just the direct child.
            console.warn("[AURA] group kill failed, killing direct child:", err);
        }
    }
    backendProcess.kill();
}
const FULL_W = 900, FULL_H = 900;
const COMP_W = 120, COMP_H = 120, COMP_MARGIN = 15;
const ANIM_MS = 380;
function getFullBounds() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;
    return {
        x: Math.round((width - FULL_W) / 2),
        y: Math.round((height - FULL_H) / 2),
        width: FULL_W,
        height: FULL_H,
    };
}
function getCompactBounds() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;
    return {
        x: width - COMP_W - COMP_MARGIN,
        y: height - COMP_H - COMP_MARGIN,
        width: COMP_W,
        height: COMP_H,
    };
}
function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}
let animTimer = null;
function animateBounds(win, from, to, ms, onDone) {
    if (animTimer) {
        clearInterval(animTimer);
        animTimer = null;
    }
    const TICK = 16, steps = Math.max(1, Math.round(ms / TICK));
    let step = 0;
    animTimer = setInterval(() => {
        step++;
        const t = easeInOut(Math.min(step / steps, 1));
        win.setBounds({
            x: Math.round(from.x + (to.x - from.x) * t),
            y: Math.round(from.y + (to.y - from.y) * t),
            width: Math.round(from.width + (to.width - from.width) * t),
            height: Math.round(from.height + (to.height - from.height) * t),
        });
        if (step >= steps) {
            clearInterval(animTimer);
            animTimer = null;
            win.setBounds(to);
            onDone?.();
        }
    }, TICK);
}
// NOTE ON RESTORE STRATEGY:
// This widget used to poll `xdotool getactivewindow` to guess when the desktop
// became active and then auto-restore. That is fundamentally broken on Wayland:
// AURA runs as an XWayland (X11) window, but the apps you click are native
// Wayland windows that XWayland's X server cannot see — so xdotool returns
// empty ("desktop active") the instant you focus another app, snapping the orb
// straight back. We now drive compact/restore purely off Electron's own
// window 'blur'/'focus' events, which the compositor delivers correctly on both
// X11 and Wayland. No external tools required.
let win;
let isCompact = false, isAnimating = false;
let savedBounds = null;
let restoreCooldownUntil = 0; // ignore blur events briefly after a restore
let compactCooldownUntil = 0; // ignore focus events briefly after compacting
function enterCompact() {
    if (isCompact || isAnimating)
        return;
    console.log("[AURA] enterCompact");
    isCompact = true;
    isAnimating = true;
    // The window can transiently gain focus while it raises to always-on-top;
    // ignore focus-driven restore until the compact animation has settled.
    compactCooldownUntil = Date.now() + ANIM_MS + 500;
    savedBounds = win.getBounds();
    win.setAlwaysOnTop(true, "screen-saver");
    // The cube is CLICKABLE — clicking it focuses the window, which restores the
    // orb via the 'focus' handler. Works on Wayland, where xdotool cannot see
    // native-Wayland windows.
    win.setIgnoreMouseEvents(false);
    win.webContents.send("enter-compact");
    animateBounds(win, savedBounds, getCompactBounds(), ANIM_MS, () => {
        isAnimating = false;
    });
}
function exitCompact() {
    if (!isCompact || isAnimating)
        return;
    console.log("[AURA] exitCompact");
    isCompact = false;
    isAnimating = true;
    const from = win.getBounds();
    const to = savedBounds ?? getFullBounds();
    win.webContents.send("enter-full"); // React starts fade-in immediately
    animateBounds(win, from, to, ANIM_MS, () => {
        isAnimating = false;
        win.setAlwaysOnTop(false);
        restoreCooldownUntil = Date.now() + 700; // don't let the focus() below bounce us back to compact
        win.focus(); // AURA becomes active window → future blur events fire
    });
}
app.whenReady().then(() => {
    console.log("[AURA] main build: cube-fix-v3 (focus-based restore, Wayland-safe)");
    startBackend();
    const bounds = getFullBounds();
    win = new BrowserWindow({
        ...bounds,
        transparent: true,
        frame: false,
        backgroundColor: "#00000000",
        alwaysOnTop: false,
        skipTaskbar: true,
        hasShadow: false,
        resizable: false,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true,
        },
    });
    win.loadFile(path.join(__dirname, "../dist/index.html"));
    // COMPACT trigger: another application became the active OS window.
    win.on("blur", () => {
        if (isCompact || isAnimating)
            return;
        if (Date.now() < restoreCooldownUntil)
            return; // ignore the blur that trails a restore
        console.log("[AURA] blur → compact");
        enterCompact();
    });
    // RESTORE trigger: AURA regained focus (cube clicked, or alt-tabbed back).
    // This is the Wayland-safe replacement for xdotool desktop polling — the
    // compositor delivers focus reliably regardless of X11/Wayland.
    win.on("focus", () => {
        if (!isCompact || isAnimating)
            return;
        if (Date.now() < compactCooldownUntil)
            return; // ignore the transient focus during compact
        console.log("[AURA] focus → restore");
        exitCompact();
    });
});
ipcMain.on("manual-compact", () => {
    if (!isCompact && !isAnimating)
        enterCompact();
});
ipcMain.on("restore-full", () => {
    if (isCompact && !isAnimating)
        exitCompact();
}); // cube click → restore
ipcMain.on("compact-ready", () => { });
ipcMain.on("set-click-through", () => { });
app.on("window-all-closed", () => app.quit());
// Make sure the spawned Python backend never outlives the Electron app.
app.on("before-quit", stopBackend);
process.on("exit", stopBackend);
