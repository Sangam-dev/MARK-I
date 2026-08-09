# Plan: AURA → Transparent Desktop Widget

## Context
The user wants to convert the current full-screen web app (1300×900, black background) into a compact transparent desktop widget — a floating circular orb that sits on the Windows desktop with no window chrome, no black background (the OS wallpaper shows through the transparent areas), and only the essential elements: the animated central orb (AI states) and a minimal clock display.

Reference image shows the JARVIS/LINKS launcher running as a native-looking transparent widget directly on the desktop wallpaper. This requires **Electron** — a browser cannot expose the OS desktop underneath its window.

The previous Electron attempt (in the conversation history) failed because:
- VW/VH were changed incorrectly, breaking the polar coordinate layout
- Electron window size was mismatched to the SVG canvas size
- No proper transparent-background CSS was applied

This plan fixes all of those issues with a methodical approach.

---

## What Changes

### 1. Compact the SVG canvas (App.tsx)
Change the coordinate system from full-screen to a small square:

| Constant | Old   | New  |
|----------|-------|------|
| VW       | 1300  | 520  |
| VH       | 900   | 520  |
| CX       | 650   | 260  |
| CY       | 450   | 260  |
| CORE     | 90    | 90 (unchanged) |

The rings (r: 102–188) and menu labels (r: 248) all fit comfortably inside a 260px radius canvas. Nothing else in the geometry needs to change — every `polar()` call references CX/CY, so updating those two constants recenters everything automatically.

### 2. Remove full-screen-only decorations (App.tsx)
Delete/simplify elements that only made sense at 1300×900:
- The long bottom status bar line + coordinates text
- The wide 32px corner bracket decorations (keep them but shrink to 16px and tuck inside the widget)
- The top-right/top-left HUD label blocks → replace with a single compact clock in the top arc area of the widget

Keep everything else intact: rings, orb, waveform, scan beam, speaking bars, orange arc, menu labels (hover-for-0.5s still works), state controls at the bottom.

### 3. Transparent background (App.tsx)
```tsx
// Root div: was background:"#050505"
background: "transparent"
```
The SVG already uses `radialGradient` and `rgba` fills — most of the "black" area is actually semi-transparent. The `bg-g` gradient goes to 0 opacity at its edges. The core-g gradient uses rgba stops. Once the HTML background is transparent and Electron is configured correctly, the OS desktop shows through.

Add a compact clock inside the SVG (e.g., `polar(230, 315)` — bottom-right area):
```tsx
<text x={...} y={...} fill={CYN} fontSize="8" fontFamily="'Share Tech Mono',monospace" opacity=".5">{clock}</text>
```

### 4. Draggable widget (App.tsx)
Add CSS to make the SVG area draggable (Electron interprets this):
```tsx
<svg style={{ WebkitAppRegion: "drag" as any, ... }}>
```
Interactive elements (menu items, state buttons) need `-webkit-app-region: no-drag` via a CSS class override so clicks still register.

### 5. Electron main process — `electron/main.js`

**Linux note:** Transparent windows on Linux require a compositor (X11: picom/compton/GNOME/KDE compositors). Without one, transparent areas render black. This is a system requirement, not a code limitation — the vast majority of modern Linux desktops (GNOME, KDE Plasma, XFCE+picom) have compositing on by default.

**Wayland note:** Transparent frameless windows on Wayland have limited support in Electron. We pass `--ozone-platform-hint=auto` so the app auto-selects X11 (via XWayland) where transparency works reliably.

```js
const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')

// Force X11 backend on Linux for reliable transparency (works under XWayland too)
if (process.platform === 'linux') {
  app.commandLine.appendSwitch('ozone-platform-hint', 'x11')
  app.commandLine.appendSwitch('enable-transparent-visuals')
}

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 520,
    height: 520,
    transparent: true,
    frame: false,
    backgroundColor: '#00000000',
    alwaysOnTop: false,
    skipTaskbar: true,
    hasShadow: false,
    resizable: false,
    // Linux: 'toolbar' type keeps it off the taskbar but above the desktop layer
    type: process.platform === 'linux' ? 'toolbar' : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  })
  win.loadFile('dist/index.html')
})
```

### 6. Electron preload — `electron/preload.js`
```js
const { contextBridge, ipcRenderer } = require('electron')
contextBridge.exposeInMainWorld('electronAPI', {
  setClickThrough: (ignore) => ipcRenderer.send('set-click-through', ignore),1. Updated 
package.json
Added Flags: Appended --no-sandbox (bypasses Linux root permission requirement for Chromium's sandbox helper) and --gtk-version=3 (resolves GTK 2/3 and GTK 4 library conflicts on Linux) to the "electron" start script:

"electron": "vite build && tsc --project tsconfig.electron.json && electron . --no-sandbox --gtk-version=3"

Added Dependency: Added "typescript": "^5.0.0" under "devDependencies" to resolve the compile-time sh: 1: tsc: not found error.
2. Configured 
pnpm-workspace.yaml
Allowed Build Scripts: Changed placeholder values to true to authorize postinstall build scripts, resolving the [ERR_PNPM_IGNORED_BUILDS] block during dependency installation:allowBuilds:
  '@tailwindcss/oxide': true
  electron: true
  esbuild: true
  
  3. Restructured Draggable Regions in 
App.tsx
Fixed Non-Responsive UI: Removed the className="drag-region" from the main <svg> element (which covers the entire window and blocked click events from reaching the React application) and changed it to className="no-drag".
Created Drag Handle: Added a dedicated, transparent absolute HTML div overlaying the central core orb (160px wide) to act as the window drag handle:

<div className="drag-region" style={{
  position: "absolute",
  left: "50%",
  top: "50%",
  width: "160px",
  height: "160px",
  transform: "translate(-50%, -50%)",
  borderRadius: "50%",
  zIndex: 5,
  cursor: "move",
}} />
})
```

Add IPC handler in main.js:
```js
ipcMain.on('set-click-through', (event, ignore) => {
  win.setIgnoreMouseEvents(ignore, { forward: true })
})
```

In App.tsx, call `window.electronAPI?.setClickThrough(true)` when `astate === 'idle'` and `false` otherwise (so the widget is click-through when idle, interactive when active).

### 7. Vite config update — `vite.config.ts`
Add `base: './'` so asset paths are relative (required for Electron's `file://` protocol):
```ts
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  // ... rest unchanged
})
```

### 8. Package.json additions
```json
{
  "main": "electron/main.js",
  "scripts": {
    "build": "vite build",
    "electron": "electron .",
    "dist": "vite build && electron .",
    "package:linux": "vite build && electron-builder --linux AppImage deb",
    "package:win": "vite build && electron-builder --win"
  },
  "build": {
    "appId": "com.aura.widget",
    "productName": "AURA",
    "files": ["dist/**/*", "electron/**/*"],
    "linux": { "target": ["AppImage", "deb"], "category": "Utility" },
    "win":   { "target": ["nsis"] }
  }
}
```
Install dependencies:
```bash
pnpm add -D electron electron-builder
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/app/App.tsx` | Modify — VW/VH/CX/CY, transparent bg, remove full-screen decorations, compact clock, drag region |
| `vite.config.ts` | Modify — add `base: './'` |
| `package.json` | Modify — add `main`, electron scripts |
| `electron/main.js` | Create — transparent frameless BrowserWindow |
| `electron/preload.js` | Create — exposes click-through IPC |

---

## Platform Notes Summary

| Feature | Linux (X11 + compositor) | Linux (no compositor) | Windows |
|---|---|---|---|
| Transparent bg | ✅ Works | ❌ Black areas | ✅ Works |
| Frameless window | ✅ | ✅ | ✅ |
| Drag widget | ✅ | ✅ | ✅ |
| Skip taskbar | ✅ | ✅ | ✅ |
| Click-through idle | ✅ | ✅ | ✅ |

**Linux requirement:** A compositor must be running. GNOME Shell, KDE Plasma, and XFCE+picom all satisfy this. This is true for essentially every modern Linux desktop out of the box.

## Verification
1. `pnpm add -D electron electron-builder` — install deps
2. `pnpm run build` — Vite builds to `dist/`
3. `pnpm run electron` — Electron opens a 520×520 transparent frameless window over the desktop
4. On Linux: confirm wallpaper is visible through transparent areas (requires compositor)
5. Confirm orb animations, state controls, and hover-menu all work
6. Confirm window is draggable by clicking and dragging the SVG ring area
7. `pnpm run package:linux` — produces `.AppImage` and `.deb` in `dist/` for distribution
