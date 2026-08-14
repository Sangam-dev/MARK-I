/**
 * Thin WebSocket client for the KANCHA backend (`api/server.py`, `/ws`).
 *
 * Wire format (see answers/integration_plan.md section 3.4 and
 * answers/guide.md):
 *
 *   Client -> Server: { type: "user_text" | "retry_last" | "ping" | "toggle_listen", payload, session_id }
 *   Server -> Client: { type: "state" | "transcript" | "response" | "response_partial" | "task_update" | "error" | "pong", payload, session_id }
 *
 * This module is intentionally the ONLY place in the frontend that touches
 * the raw `WebSocket` API — App.tsx and friends subscribe via `wsClient.on(...)`
 * and never construct a WebSocket directly. No Electron IPC bridge is
 * needed here: the renderer can reach `ws://127.0.0.1:8765` directly even
 * with `contextIsolation: true`, because WebSocket/fetch are standard
 * browser APIs, not Node/Electron internals.
 */

export type ServerMessageType =
  | "state"
  | "transcript"
  | "response"
  | "response_partial"
  | "task_update"
  | "error"
  | "pong"
  // Pseudo-events emitted locally by this client (not part of the wire
  // format) so consumers can react to connect/disconnect uniformly via `on`.
  | "_open"
  | "_close";

export interface ServerMessage<T = unknown> {
  type: ServerMessageType;
  payload: T;
  session_id: string;
}

type Listener<T = unknown> = (payload: T, message: ServerMessage<T>) => void;

const DEFAULT_WS_URL = "ws://127.0.0.1:8765/ws";

function resolveWsUrl(): string {
  const env = (import.meta as unknown as { env?: Record<string, string> }).env;
  return env?.VITE_BACKEND_WS_URL || DEFAULT_WS_URL;
}

const MAX_RECONNECT_DELAY_MS = 10_000;
const INITIAL_RECONNECT_DELAY_MS = 1_000;

class WsClient {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<Listener>>();
  private reconnectDelay = INITIAL_RECONNECT_DELAY_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private shouldReconnect = true;
  private _connected = false;

  get isConnected(): boolean {
    return this._connected;
  }

  connect(): void {
    this.shouldReconnect = true;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const ws = new WebSocket(resolveWsUrl());
    this.ws = ws;

    ws.onopen = () => {
      this._connected = true;
      this.reconnectDelay = INITIAL_RECONNECT_DELAY_MS;
      this.emit("_open", {});
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      let message: ServerMessage;
      try {
        message = JSON.parse(event.data);
      } catch {
        console.warn("[wsClient] received non-JSON message, ignoring:", event.data);
        return;
      }
      this.emit(message.type, message.payload, message);
    };

    ws.onclose = () => {
      this._connected = false;
      this.emit("_close", {});
      if (this.shouldReconnect) {
        this.scheduleReconnect();
      }
    };

    ws.onerror = () => {
      // onclose fires right after onerror for browser WebSockets; the
      // reconnect loop lives there so we don't double-schedule here.
      ws.close();
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, MAX_RECONNECT_DELAY_MS);
  }

  /** Subscribe to a server message type (or "_open"/"_close"). Returns an unsubscribe function. */
  on<T = unknown>(type: ServerMessageType, cb: Listener<T>): () => void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    const set = this.listeners.get(type)!;
    set.add(cb as Listener);
    return () => set.delete(cb as Listener);
  }

  private emit(type: string, payload: unknown, message?: ServerMessage): void {
    this.listeners.get(type)?.forEach((cb) => {
      try {
        cb(payload, (message ?? { type, payload, session_id: "default" }) as ServerMessage);
      } catch (err) {
        console.error(`[wsClient] listener for "${type}" threw:`, err);
      }
    });
  }

  send(type: "user_text" | "retry_last" | "ping" | "toggle_listen", payload: Record<string, unknown> = {}, sessionId = "default"): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn(`[wsClient] not connected, dropped "${type}"`);
      return false;
    }
    this.ws.send(JSON.stringify({ type, payload, session_id: sessionId }));
    return true;
  }
}

/** Module-level singleton — one Electron renderer, one backend connection. */
export const wsClient = new WsClient();
