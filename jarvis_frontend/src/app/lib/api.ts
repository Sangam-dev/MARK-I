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

async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${resolveHttpUrl()}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

/**
 * FastAPI reports validation and business errors in a `detail` field. Surfacing
 * that instead of a bare status code is what lets the upload widget say
 * "Unsupported file type. Supported formats: .pdf, .txt, .md, .docx" rather
 * than "415".
 */
async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${resolveHttpUrl()}${path}`, { method: "POST", body: form });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body — keep the status code.
    }
    throw new Error(detail);
  }
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
  rag_enabled: boolean;
  rag_available: boolean;
  rag_documents: number;
  ws_clients: number;
  uptime_seconds: number;
}

export interface ActionResult {
  success: boolean;
  message: string;
}

// ── RAG (long-term semantic memory) ──────────────────────────────────────────

export interface RAGUploadResult {
  success: boolean;
  filename: string;
  message: string;
  document_id: string;
  title: string;
  chunks_indexed: number;
  chunks_skipped: number;
  error: string;
  metadata: Record<string, unknown>;
}

export interface RAGDocument {
  id: string;
  title: string;
  source: string;
  doc_type: string;
  created_at: string;
  updated_at: string;
  chunk_count: number;
  metadata: Record<string, unknown>;
}

export interface RAGStats {
  ready: boolean;
  enabled: boolean;
  embedder: string;
  store: string;
  dimensions: number;
  documents: number;
  chunks: number;
  top_k: number;
  similarity_threshold: number;
  chunk_size: number;
  chunk_overlap: number;
  supported_extensions: string[];
  max_upload_bytes: number;
  error: string;
}

export interface RAGSearchHit {
  title: string;
  type: string;
  score: number;
  content: string;
  source: string;
  document_id: string;
  chunk_id: string;
  metadata: Record<string, unknown>;
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

  // RAG — the upload pipeline is independent of the conversation pipeline;
  // these calls never touch the WebSocket.
  uploadDocument: (file: File, title?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return postForm<RAGUploadResult>("/api/rag/upload", form);
  },
  getRagDocuments: () => getJson<RAGDocument[]>("/api/rag/documents"),
  deleteRagDocument: (id: string) =>
    deleteJson<{ document_id: string; title: string; chunks_removed: number }>(
      `/api/rag/documents/${encodeURIComponent(id)}`,
    ),
  getRagStats: () => getJson<RAGStats>("/api/rag/stats"),
  searchRag: (q: string, topK?: number) =>
    getJson<{ query: string; count: number; results: RAGSearchHit[] }>(
      `/api/rag/search?q=${encodeURIComponent(q)}${topK ? `&top_k=${topK}` : ""}`,
    ),
};
