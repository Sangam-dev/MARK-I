/**
 * REST client for the KANCHA backend (`api/server.py`, routes under `/api`).
 *
 * Every function here maps 1:1 to an endpoint documented in
 * answers/integration_plan.md section 3.5 and answers/guide.md. Add new
 * functions here (not ad-hoc `fetch` calls scattered through components)
 * when wiring a new panel to real backend data.
 */

const DEFAULT_HTTP_URL = "http://127.0.0.1:8765";

function resolveHttpUrl(): string {
  const env = (import.meta as unknown as { env?: Record<string, string> }).env;
  return env?.VITE_BACKEND_HTTP_URL || DEFAULT_HTTP_URL;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${resolveHttpUrl()}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${resolveHttpUrl()}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PUT ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export interface FactOut {
  id: string;
  key: string;
  value: string;
  created_at: string;
  updated_at: string;
}

export interface HistoryTurn {
  role: string;
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface SettingsModel {
  tts_enabled: boolean;
  voice_mode: boolean;
  session_id: string;
}

export interface HealthInfo {
  status: string;
  session_id: string;
  tts_enabled: boolean;
  llm_available: boolean;
  ws_clients: number;
  uptime_seconds: number;
}

export interface ActionResult {
  success: boolean;
  message: string;
}

export const api = {
  getHealth: () => getJson<HealthInfo>("/api/health"),
  getFacts: () => getJson<FactOut[]>("/api/memory/facts"),
  getHistory: (limit = 20) => getJson<HistoryTurn[]>(`/api/history?limit=${limit}`),
  getSettings: () => getJson<SettingsModel>("/api/settings"),
  updateSettings: (settings: SettingsModel) => putJson<SettingsModel>("/api/settings", settings),
  getAlarms: () => getJson<ActionResult>("/api/alarms"),
  listFiles: (path = "desktop") =>
    getJson<{ path: string; message: string }>(`/api/files?path=${encodeURIComponent(path)}`),
  getWeather: (city: string) =>
    getJson<ActionResult>(`/api/weather?city=${encodeURIComponent(city)}`),
};
