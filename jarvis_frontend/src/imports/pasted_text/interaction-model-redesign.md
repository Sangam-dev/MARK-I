The current implementation is close, but I want to redesign the interaction model. Do not patch the existing behavior—replace it with the following architecture.

## New Interaction Model

The assistant should no longer rely on clicking the compact companion to restore the full orb.

Everything should be automatic and driven by the user's workspace (Desktop vs. Application).

The assistant should intelligently switch between two visual states:

1. Full Orb
2. Compact Companion

The transition should always feel like the same object changing form.

---

# State 1 — Desktop Mode

Whenever the user is on the Windows/Linux desktop (no application window is currently active in the foreground), the assistant should automatically expand into the existing Full Orb.

No user interaction is required.

The current orb UI should appear exactly as it does now.

Preserve:

- Existing orb
- Existing animations
- Existing effects
- Existing particles
- Existing conversation state

Do not redesign anything.

---

# State 2 — Application Mode

Whenever the user opens, switches to, or focuses any application window, the assistant should automatically transition into the compact companion.

Examples:

- Chrome
- VS Code
- Terminal
- File Explorer
- Spotify
- Discord
- Steam
- Games

As soon as one of these windows becomes the active window, begin the transition immediately.

No timers.

No inactivity detection.

No clicking.

This must be completely event-driven.

---

# Transition Animation

The transition should happen simultaneously at both the BrowserWindow level and the UI level.

Sequence:

1. Fade secondary UI elements.
2. Compress the orb.
3. Shrink proportionally.
4. Move diagonally toward the bottom-right corner.
5. Resize the BrowserWindow at the same time.
6. Morph into the compact companion.
7. Dock with a subtle spring animation.

The user should perceive this as one object changing form.

Never destroy or recreate the BrowserWindow.

Never reload React.

Never lose assistant state.

---

# Compact Companion Redesign

Replace the current compact icon completely.

I no longer want a circular icon.

Instead, create a futuristic rotating cube.

Visual style:

- Transparent holographic cube
- Cyan / blue energy glow
- Minimal sci-fi aesthetic
- Floating in space
- Semi-transparent edges
- Bright illuminated corners
- Thin glowing outlines
- Subtle animated inner geometry
- Energy flowing through the edges

The cube should feel like the compressed processing core of the assistant.

---

# Cube Orientation

The cube should NOT sit flat.

It should be permanently tilted in an isometric perspective.

Specifically:

- Rotate approximately 35–45° around the Y-axis.
- Rotate approximately 20–30° around the X-axis.
- This should expose three visible faces at all times.
- The cube should appear to float in 3D space.

The overall appearance should resemble a premium holographic data cube rather than a simple geometric cube.

---

# Cube Idle Animation

The cube should always appear alive.

Idle animation:

- Continuous slow rotation around its vertical axis.
- Maintain the tilted orientation while rotating.
- Gentle floating motion.
- Soft breathing glow.
- Animated light moving through the cube edges.
- Small energy pulses from the center.
- Optional subtle particle effects around the cube.

Animations should remain smooth and lightweight.

---

# Window Behavior

The compact cube should live inside a BrowserWindow that is approximately 70–80 px.

The BrowserWindow should:

- Always remain on top.
- Be transparent.
- Be frameless.
- Be hidden from the taskbar.
- Be hidden from Alt+Tab.
- Be positioned relative to the physical display, not the React canvas.
- Dock with approximately 20–24 px margins from the bottom-right edge of the active monitor.

The entire BrowserWindow should resize with the transition.

No invisible transparent window should remain on the desktop.

---

# Automatic Restoration

The compact cube should NOT be clickable to restore the assistant.

Instead, restoration is fully automatic.

Whenever the user returns to the desktop (no application window is currently active in the foreground), the assistant should:

1. Detect that the desktop has become active.
2. Animate the cube away from the bottom-right corner.
3. Enlarge while moving.
4. Resize the BrowserWindow simultaneously.
5. Reconstruct the existing orb.
6. Restore the full interface.
7. Resume every existing animation.

No user interaction should be required.

---

# State Machine

The behavior should simply be:

Desktop Active
    ↓
Full Orb

Application Gains Focus
    ↓
Transition
    ↓
Compact Cube

Desktop Becomes Active Again
    ↓
Transition
    ↓
Full Orb

There should be no additional states, timers, or manual interactions.

---

# Architecture

Review the existing implementation and refactor it if necessary.

The BrowserWindow should control:

- Position
- Size
- Docking
- Screen-relative coordinates
- Workspace transitions

The React application should control:

- Orb rendering
- Cube rendering
- Morph animations
- Visual effects
- UI state

Desktop behavior must not be implemented solely with CSS transforms.

Use Electron's native APIs and OS-level workspace/window focus detection wherever appropriate.

---

# Goal

The final experience should feel like a native AI operating system companion.

When the user is on the desktop, the assistant naturally expands into the full orb.

The moment the user begins working in any application, the orb gracefully compresses into a tilted, holographic rotating cube that docks in the bottom-right corner of the screen.

When the user returns to the desktop, the cube automatically unfolds back into the full assistant.

The experience should feel seamless, intelligent, cinematic, and continuous, as though a single living AI entity is adapting its form based on the user's workspace.