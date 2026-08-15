"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('electronAPI', {
    onEnterCompact: (cb) => {
        electron_1.ipcRenderer.on('enter-compact', cb);
        return () => electron_1.ipcRenderer.removeListener('enter-compact', cb);
    },
    onEnterFull: (cb) => {
        electron_1.ipcRenderer.on('enter-full', cb);
        return () => electron_1.ipcRenderer.removeListener('enter-full', cb);
    },
    // Ask the main process to restore the full orb (fired when the cube is clicked).
    requestRestore: () => electron_1.ipcRenderer.send('restore-full'),
    // Ask the main process to stop the backend and exit (fired after a spoken
    // "quit"/"exit" confirmation, and by the tray's "Quit AURA" via main directly).
    requestQuit: () => electron_1.ipcRenderer.send('quit-app'),
});
