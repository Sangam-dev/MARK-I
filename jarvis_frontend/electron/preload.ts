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
})
