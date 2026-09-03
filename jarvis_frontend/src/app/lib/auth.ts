export interface ElectronAuthAPI {
  onEnterCompact?: (cb: () => void) => () => void;
  onEnterFull?: (cb: () => void) => () => void;
  requestRestore?: () => void;
  requestQuit?: (confirmation?: boolean) => void;
  getAuthToken?: () => Promise<string>;
}

declare global {
  interface Window {
    electronAPI?: ElectronAuthAPI;
  }
}

export async function getAuthToken(): Promise<string> {
  const fromElectron = await window.electronAPI?.getAuthToken?.();
  return fromElectron || import.meta.env.VITE_KANCHA_AUTH_TOKEN || "";
}
