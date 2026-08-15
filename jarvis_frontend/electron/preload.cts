import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  onEnterCompact: (cb: () => void) => {
    ipcRenderer.on('enter-compact', cb)
    return () => ipcRenderer.removeListener('enter-compact', cb)
  },
  onEnterFull: (cb: () => void) => {
    ipcRenderer.on('enter-full', cb)
    return () => ipcRenderer.removeListener('enter-full', cb)
  },
  // Ask the main process to restore the full orb (fired when the cube is clicked).
  requestRestore: () => ipcRenderer.send('restore-full'),
  // Ask the main process to stop the backend and exit (fired after a spoken
  // "quit"/"exit" confirmation, and by the tray's "Quit AURA" via main directly).
  requestQuit: () => ipcRenderer.send('quit-app'),
})
