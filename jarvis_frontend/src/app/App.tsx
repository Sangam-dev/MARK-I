import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { wsClient } from "./lib/wsClient";
import {
    api,
    type FactOut,
    type HistoryTurn,
    type SettingsModel,
    type HealthInfo,
    type RAGDocument,
    type RAGStats,
} from "./lib/api";

declare global {
    interface Window {
        electronAPI?: {
            onEnterCompact: (cb: () => void) => () => void;
            onEnterFull: (cb: () => void) => () => void;
            requestRestore: () => void;
        };
    }
}

type AState = "sleeping" | "idle" | "listening" | "thinking" | "speaking";
type PanelId =
    | "settings"
    | "history"
    | "memory"
    | "tools"
    | "automation"
    | "calendar"
    | "developer"
    | "files";
type Mode = "full" | "compact";

// ── Canvas ────────────────────────────────────────────────────────────────────
const VW = 520,
    VH = 520,
    CX = 260,
    CY = 260,
    CORE = 80;
const CYN = "#00e5ff",
    CYN2 = "#00b8d4";
const ORG = "#ff9800",
    GRN = "#00ff88",
    RED = "#ff4444";

// ── Geometry ──────────────────────────────────────────────────────────────────
function polar(r: number, deg: number): [number, number] {
    const a = (deg - 90) * (Math.PI / 180);
    return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}
function sector(r1: number, r2: number, a1: number, a2: number): string {
    const [x1, y1] = polar(r1, a1),
        [x2, y2] = polar(r1, a2);
    const [x3, y3] = polar(r2, a2),
        [x4, y4] = polar(r2, a1);
    const lg = a2 - a1 > 180 ? 1 : 0;
    return `M${x1} ${y1} A${r1} ${r1} 0 ${lg} 1 ${x2} ${y2} L${x3} ${y3} A${r2} ${r2} 0 ${lg} 0 ${x4} ${y4}Z`;
}
function arcOnly(r: number, a1: number, a2: number): string {
    const [x1, y1] = polar(r, a1),
        [x2, y2] = polar(r, a2);
    const lg = a2 - a1 > 180 ? 1 : 0;
    return `M${x1} ${y1} A${r} ${r} 0 ${lg} 1 ${x2} ${y2}`;
}

// ── Config ────────────────────────────────────────────────────────────────────
const RINGS = [
    { r: 92, w: 1.5, d: "3 6", o: 0.7, s: 20, cw: true },
    { r: 103, w: 1.0, d: "1 8", o: 0.4, s: 28, cw: false },
    { r: 115, w: 1.5, d: "8 3", o: 0.55, s: 40, cw: true },
    { r: 127, w: 0.8, d: "2 5", o: 0.3, s: 50, cw: false },
    { r: 140, w: 1.2, d: "6 5", o: 0.45, s: 30, cw: true },
    { r: 152, w: 0.6, d: "1 4", o: 0.25, s: 70, cw: false },
    { r: 165, w: 1.0, d: "4 3", o: 0.5, s: 55, cw: true },
];

const MENUS: { id: PanelId; label: string; angle: number }[] = [
    { id: "developer", label: "DEV", angle: 0 },
    { id: "settings", label: "SETTINGS", angle: 45 },
    { id: "tools", label: "TOOLS", angle: 90 },
    { id: "history", label: "HISTORY", angle: 135 },
    { id: "memory", label: "MEMORY", angle: 180 },
    { id: "files", label: "FILES", angle: 225 },
    { id: "calendar", label: "CALENDAR", angle: 270 },
    { id: "automation", label: "AUTO", angle: 315 },
];

const STATE_LBL: Record<AState, string> = {
    sleeping: "SAY HEY JARVIS",
    idle: "IDLE",
    listening: "LISTENING...",
    thinking: "THINKING...",
    speaking: "SPEAKING...",
};

const PANEL_POS: Record<PanelId, React.CSSProperties> = {
    developer: { top: "14px", left: "50%", transform: "translateX(-50%)" },
    settings: { top: "14px", right: "14px" },
    tools: { top: "50%", right: "14px", transform: "translateY(-50%)" },
    history: { bottom: "14px", right: "14px" },
    memory: { bottom: "14px", left: "50%", transform: "translateX(-50%)" },
    files: { bottom: "14px", left: "14px" },
    calendar: { top: "50%", left: "14px", transform: "translateY(-50%)" },
    automation: { top: "14px", left: "14px" },
};

const PANEL_VARIANTS = {
    enter: (angle: number) => ({
        opacity: 0,
        x: Math.sin((angle * Math.PI) / 180) * 30,
        y: -Math.cos((angle * Math.PI) / 180) * 30,
        scale: 0.84,
        filter: "blur(8px)",
    }),
    center: {
        opacity: 1,
        x: 0,
        y: 0,
        scale: 1,
        filter: "blur(0px)",
        transition: { type: "spring" as const, stiffness: 380, damping: 28 },
    },
    exit: (angle: number) => ({
        opacity: 0,
        x: -Math.sin((angle * Math.PI) / 180) * 20,
        y: Math.cos((angle * Math.PI) / 180) * 20,
        scale: 0.91,
        filter: "blur(6px)",
        transition: { duration: 0.18, ease: "easeIn" as const },
    }),
};

// ── Panel content ────────────────────────────────────────────────────────────────────
//
// SETTINGS / HISTORY / MEMORY / AUTOMATION / FILES / DEVELOPER are backed by
// real backend data once it loads (see the panel-data fetch effect in App()).
// Until it loads (or if the backend is offline), each falls back to the
// original static preview content so the widget still looks alive.
// TOOLS and CALENDAR have no backend module yet (see answers/guide.md,
// "Known gaps") and intentionally stay mocked.
export interface PanelData {
    facts: FactOut[] | null;
    history: HistoryTurn[] | null;
    settingsData: SettingsModel | null;
    alarmsMessage: string | null;
    filesMessage: string | null;
    health: HealthInfo | null;
    wsLatencyMs: number | null;
    onToggleTts: () => void;
    onToggleVoiceMode: () => void;
    // RAG knowledge base — backs the upload widget in the FILES panel.
    ragDocuments: RAGDocument[] | null;
    ragStats: RAGStats | null;
    onRagChanged: () => void;
}

// ── RAG knowledge base widget ────────────────────────────────────────────────
//
// Drives the upload pipeline (POST /api/rag/upload), which is completely
// independent of the conversation pipeline — no WebSocket traffic, no bus
// events. Uploaded documents are chunked, embedded and indexed server-side;
// the assistant then retrieves from them automatically when a question
// warrants it (see memory/rag/router.py for the retrieve/skip decision).
function RagKnowledgeBase({
    documents,
    stats,
    onChanged,
}: {
    documents: RAGDocument[] | null;
    stats: RAGStats | null;
    onChanged: () => void;
}) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(
        null,
    );
    const [dragging, setDragging] = useState(false);

    const accept = (stats?.supported_extensions ?? [".pdf", ".txt", ".md", ".docx"]).join(",");
    const available = stats?.ready !== false;

    const upload = async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        setBusy(true);
        setStatus(null);

        let indexed = 0;
        const failures: string[] = [];
        for (const file of Array.from(files)) {
            try {
                const result = await api.uploadDocument(file);
                indexed += result.chunks_indexed;
            } catch (err) {
                failures.push(
                    `${file.name}: ${err instanceof Error ? err.message : String(err)}`,
                );
            }
        }

        setBusy(false);
        setStatus(
            failures.length === 0
                ? { ok: true, text: `Indexed ${indexed} chunk(s) from ${files.length} file(s).` }
                : { ok: false, text: failures[0] },
        );
        onChanged();
        if (inputRef.current) inputRef.current.value = "";
    };

    const remove = async (id: string) => {
        try {
            await api.deleteRagDocument(id);
            setStatus({ ok: true, text: "Document removed." });
            onChanged();
        } catch (err) {
            setStatus({
                ok: false,
                text: err instanceof Error ? err.message : String(err),
            });
        }
    };

    return (
        <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid rgba(0,229,255,0.14)" }}>
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 8,
                }}
            >
                <span style={{ fontSize: 10, letterSpacing: 2, opacity: 0.75 }}>
                    KNOWLEDGE BASE
                </span>
                <span style={{ fontSize: 9, opacity: 0.45 }}>
                    {stats ? `${stats.documents} docs · ${stats.chunks} chunks` : "—"}
                </span>
            </div>

            {!available ? (
                <div style={{ fontSize: 11, color: ORG, opacity: 0.8 }}>
                    RAG is offline — uploads are disabled. Check the backend logs.
                </div>
            ) : (
                <>
                    <div
                        className="no-drag"
                        role="button"
                        tabIndex={0}
                        onClick={() => !busy && inputRef.current?.click()}
                        onDragOver={(e) => {
                            e.preventDefault();
                            setDragging(true);
                        }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={(e) => {
                            e.preventDefault();
                            setDragging(false);
                            if (!busy) upload(e.dataTransfer.files);
                        }}
                        style={{
                            border: `1px dashed ${dragging ? CYN : "rgba(0,229,255,0.35)"}`,
                            background: dragging ? "rgba(0,229,255,0.07)" : "transparent",
                            borderRadius: 4,
                            padding: "12px 8px",
                            textAlign: "center",
                            cursor: busy ? "progress" : "pointer",
                            fontSize: 11,
                            opacity: busy ? 0.55 : 1,
                            transition: "background .15s, border-color .15s",
                        }}
                    >
                        {busy ? (
                            <span style={{ color: CYN }}>INDEXING…</span>
                        ) : (
                            <>
                                <div style={{ color: CYN, marginBottom: 3 }}>
                                    ↑ DROP FILES OR CLICK
                                </div>
                                <div style={{ fontSize: 9, opacity: 0.45 }}>
                                    {accept.replace(/\./g, "").toUpperCase().replace(/,/g, " · ")}
                                </div>
                            </>
                        )}
                    </div>

                    <input
                        ref={inputRef}
                        type="file"
                        multiple
                        accept={accept}
                        style={{ display: "none" }}
                        onChange={(e) => upload(e.target.files)}
                    />

                    {status && (
                        <div
                            style={{
                                fontSize: 10,
                                marginTop: 7,
                                lineHeight: 1.45,
                                color: status.ok ? GRN : RED,
                            }}
                        >
                            {status.text}
                        </div>
                    )}

                    {documents && documents.length > 0 && (
                        <div style={{ marginTop: 10, maxHeight: 130, overflowY: "auto" }}>
                            {documents.map((doc) => (
                                <div
                                    key={doc.id}
                                    style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center",
                                        gap: 6,
                                        marginBottom: 6,
                                        paddingBottom: 5,
                                        borderBottom: "1px solid rgba(0,229,255,0.06)",
                                    }}
                                >
                                    <div style={{ minWidth: 0 }}>
                                        <div
                                            style={{
                                                fontSize: 11,
                                                whiteSpace: "nowrap",
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                            }}
                                            title={doc.title}
                                        >
                                            {doc.title}
                                        </div>
                                        <div style={{ fontSize: 9, opacity: 0.4 }}>
                                            {doc.doc_type} · {doc.chunk_count} chunks
                                        </div>
                                    </div>
                                    <span
                                        className="no-drag"
                                        role="button"
                                        tabIndex={0}
                                        onClick={() => remove(doc.id)}
                                        title="Remove from knowledge base"
                                        style={{
                                            cursor: "pointer",
                                            color: RED,
                                            opacity: 0.65,
                                            fontSize: 12,
                                            flexShrink: 0,
                                            padding: "0 3px",
                                        }}
                                    >
                                        ×
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}

                    {documents && documents.length === 0 && (
                        <div style={{ fontSize: 10, opacity: 0.4, marginTop: 8 }}>
                            No documents indexed yet.
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function PanelContent({ id, data }: { id: PanelId; data: PanelData }) {
    const row: React.CSSProperties = {
        display: "flex",
        justifyContent: "space-between",
        marginBottom: 8,
        paddingBottom: 7,
        borderBottom: "1px solid rgba(0,229,255,0.08)",
        fontSize: 12,
    };
    const dim: React.CSSProperties = { opacity: 0.5 };
    const val: React.CSSProperties = { color: GRN };

    switch (id) {
        case "settings": {
            const s = data.settingsData;
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        SETTINGS
                    </h3>
                    <div
                        style={{ ...row, cursor: "pointer" }}
                        className="no-drag"
                        role="button"
                        tabIndex={0}
                        onClick={data.onToggleTts}
                    >
                        <span style={dim}>TTS</span>
                        <span style={val}>
                            {s
                                ? s.tts_enabled
                                    ? "ENABLED"
                                    : "DISABLED"
                                : "ENABLED"}
                        </span>
                    </div>
                    <div
                        style={{ ...row, cursor: "pointer" }}
                        className="no-drag"
                        role="button"
                        tabIndex={0}
                        onClick={data.onToggleVoiceMode}
                    >
                        <span style={dim}>VOICE MODE</span>
                        <span style={val}>
                            {s
                                ? s.voice_mode
                                    ? "ENABLED"
                                    : "DISABLED"
                                : "DISABLED"}
                        </span>
                    </div>
                    {(
                        [
                            ["LANGUAGE", "EN-US"],
                            ["LLM PROVIDER", "Gemini"],
                            ["SESSION", s?.session_id ?? "default"],
                        ] as [string, string][]
                    ).map(([k, v]) => (
                        <div key={k} style={row}>
                            <span style={dim}>{k}</span>
                            <span style={val}>{v}</span>
                        </div>
                    ))}
                    <div style={{ fontSize: 9, opacity: 0.35, marginTop: 6 }}>
                        Click TTS / VOICE MODE to toggle — takes effect on the
                        next backend restart.
                    </div>
                </>
            );
        }
        case "history": {
            const turns = data.history;
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        HISTORY
                    </h3>
                    {turns && turns.length > 0 ? (
                        turns
                            .slice()
                            .reverse()
                            .map((t, i) => (
                                <div
                                    key={i}
                                    style={{
                                        borderLeft:
                                            "2px solid rgba(0,229,255,0.3)",
                                        paddingLeft: 8,
                                        marginBottom: 10,
                                    }}
                                >
                                    <div style={{ fontSize: 12 }}>
                                        <span
                                            style={{
                                                opacity: 0.5,
                                                marginRight: 6,
                                            }}
                                        >
                                            {t.role === "user"
                                                ? "YOU"
                                                : "JARVIS"}
                                        </span>
                                        {t.content.length > 80
                                            ? t.content.slice(0, 80) + "…"
                                            : t.content}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: 10,
                                            opacity: 0.4,
                                            marginTop: 2,
                                        }}
                                    >
                                        {new Date(
                                            t.timestamp,
                                        ).toLocaleTimeString()}
                                    </div>
                                </div>
                            ))
                    ) : turns && turns.length === 0 ? (
                        <div style={{ fontSize: 12, opacity: 0.4 }}>
                            No conversation yet this session.
                        </div>
                    ) : (
                        (
                            [
                                ["Quantum computing", "2h ago"],
                                ["Sales analysis", "5h ago"],
                                ["Code debugging", "Yesterday"],
                                ["Trip planning", "2d ago"],
                            ] as [string, string][]
                        ).map(([t, d]) => (
                            <div
                                key={t}
                                style={{
                                    borderLeft: "2px solid rgba(0,229,255,0.3)",
                                    paddingLeft: 8,
                                    marginBottom: 10,
                                }}
                            >
                                <div style={{ fontSize: 12 }}>{t}</div>
                                <div
                                    style={{
                                        fontSize: 10,
                                        opacity: 0.4,
                                        marginTop: 2,
                                    }}
                                >
                                    {d}
                                </div>
                            </div>
                        ))
                    )}
                </>
            );
        }
        case "memory": {
            const facts = data.facts;
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        MEMORY
                    </h3>
                    {facts && facts.length > 0 ? (
                        facts.map((f) => (
                            <div
                                key={f.id}
                                style={{
                                    display: "flex",
                                    gap: 7,
                                    marginBottom: 9,
                                    fontSize: 12,
                                    alignItems: "flex-start",
                                }}
                            >
                                <span style={{ color: CYN2, flexShrink: 0 }}>
                                    ▸
                                </span>
                                <span>
                                    <span style={{ opacity: 0.6 }}>
                                        {f.key}:
                                    </span>{" "}
                                    {f.value}
                                </span>
                            </div>
                        ))
                    ) : facts && facts.length === 0 ? (
                        <div style={{ fontSize: 12, opacity: 0.4 }}>
                            No stored facts yet.
                        </div>
                    ) : (
                        [
                            "Dark theme preferred",
                            "Language: English",
                            "Technology sector",
                            "Uses VS Code",
                            "Based in New York",
                        ].map((m) => (
                            <div
                                key={m}
                                style={{
                                    display: "flex",
                                    gap: 7,
                                    marginBottom: 9,
                                    fontSize: 12,
                                    alignItems: "flex-start",
                                }}
                            >
                                <span style={{ color: CYN2, flexShrink: 0 }}>
                                    ▸
                                </span>
                                <span>{m}</span>
                            </div>
                        ))
                    )}
                </>
            );
        }
        case "tools":
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        TOOLS
                    </h3>
                    {(
                        [
                            ["Web Search", true],
                            ["Code Executor", true],
                            ["File Manager", true],
                            ["Browser Ctrl", false],
                            ["Vision API", true],
                        ] as [string, boolean][]
                    ).map(([n, a]) => (
                        <div
                            key={String(n)}
                            style={{ ...row, alignItems: "center" }}
                        >
                            <span style={dim}>{String(n)}</span>
                            <span
                                style={{
                                    fontSize: 10,
                                    color: a ? GRN : RED,
                                    border: `1px solid ${a ? "rgba(0,255,136,0.3)" : "rgba(255,68,68,0.3)"}`,
                                    padding: "2px 7px",
                                    borderRadius: 2,
                                }}
                            >
                                {a ? "ON" : "OFF"}
                            </span>
                        </div>
                    ))}
                </>
            );
        case "calendar":
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 6px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        CALENDAR
                    </h3>
                    <div
                        style={{
                            fontSize: 10,
                            opacity: 0.38,
                            marginBottom: 12,
                        }}
                    >
                        {new Date().toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                        })}
                    </div>
                    {(
                        [
                            ["09:30", "Quantum Review", CYN],
                            ["11:00", "Standup", GRN],
                            ["14:00", "Demo", ORG],
                            ["16:30", "Call", CYN],
                        ] as [string, string, string][]
                    ).map(([t, e, c]) => (
                        <div
                            key={e}
                            style={{
                                display: "flex",
                                gap: 8,
                                marginBottom: 9,
                                alignItems: "center",
                            }}
                        >
                            <span
                                style={{
                                    color: c,
                                    minWidth: 38,
                                    fontSize: 11,
                                    fontWeight: 700,
                                }}
                            >
                                {t}
                            </span>
                            <span
                                style={{
                                    fontSize: 12,
                                    borderLeft: `1px solid ${c}40`,
                                    paddingLeft: 7,
                                }}
                            >
                                {e}
                            </span>
                        </div>
                    ))}
                </>
            );
        case "automation": {
            const msg = data.alarmsMessage;
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        AUTOMATION
                    </h3>
                    {msg ? (
                        <div style={{ fontSize: 12, lineHeight: 1.6 }}>
                            {msg}
                        </div>
                    ) : (
                        (
                            [
                                ["Daily Brief", "08:00", true],
                                ["Email Digest", "18:00", true],
                                ["News Summary", "07:30", false],
                                ["Health Check", "12:00", true],
                            ] as [string, string, boolean][]
                        ).map(([n, s, a]) => (
                            <div
                                key={String(n)}
                                style={{
                                    marginBottom: 10,
                                    paddingBottom: 9,
                                    borderBottom:
                                        "1px solid rgba(0,229,255,0.08)",
                                }}
                            >
                                <div
                                    style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        fontSize: 12,
                                        alignItems: "center",
                                    }}
                                >
                                    <span>{String(n)}</span>
                                    <span
                                        style={{
                                            fontSize: 10,
                                            color: a ? GRN : ORG,
                                        }}
                                    >
                                        {a ? "ON" : "OFF"}
                                    </span>
                                </div>
                                <div
                                    style={{
                                        fontSize: 10,
                                        opacity: 0.35,
                                        marginTop: 2,
                                    }}
                                >
                                    {String(s)}
                                </div>
                            </div>
                        ))
                    )}
                    <div style={{ fontSize: 9, opacity: 0.35, marginTop: 10 }}>
                        Live alarms/reminders (actions/alarms.py). Scheduled
                        digests below are illustrative — no backend module for
                        them yet.
                    </div>
                </>
            );
        }
        case "developer": {
            const h = data.health;
            const rows: [string, string, string][] = h
                ? [
                      [
                          "STATUS",
                          h.status === "ok" ? "CONNECTED" : "ERROR",
                          h.status === "ok" ? GRN : RED,
                      ],
                      [
                          "LLM",
                          h.llm_available ? "AVAILABLE" : "UNAVAILABLE",
                          h.llm_available ? GRN : RED,
                      ],
                      [
                          "TTS",
                          h.tts_enabled ? "ENABLED" : "DISABLED",
                          h.tts_enabled ? GRN : ORG,
                      ],
                      [
                          "RAG",
                          h.rag_available
                              ? `${h.rag_documents} DOCS`
                              : h.rag_enabled
                                ? "ERROR"
                                : "DISABLED",
                          h.rag_available ? GRN : h.rag_enabled ? RED : ORG,
                      ],
                      [
                          "WS LATENCY",
                          data.wsLatencyMs != null
                              ? `${data.wsLatencyMs}ms`
                              : "—",
                          CYN,
                      ],
                      ["UPTIME", `${Math.round(h.uptime_seconds)}s`, CYN2],
                  ]
                : [
                      ["STATUS", "OFFLINE", RED],
                      ["LLM", "—", CYN],
                      ["WS LATENCY", "—", CYN],
                      ["VERSION", "v2.6.3", CYN2],
                  ];
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        DEVELOPER
                    </h3>
                    {rows.map(([k, v, c]) => (
                        <div key={k} style={row}>
                            <span style={dim}>{k}</span>
                            <span style={{ color: c }}>{v}</span>
                        </div>
                    ))}
                </>
            );
        }
        case "files": {
            const msg = data.filesMessage;
            return (
                <>
                    <h3
                        style={{
                            margin: "0 0 14px",
                            fontSize: 11,
                            letterSpacing: 3,
                            color: "white",
                            fontWeight: 700,
                        }}
                    >
                        FILES
                    </h3>
                    {msg ? (
                        <div
                            style={{
                                fontSize: 11.5,
                                lineHeight: 1.6,
                                whiteSpace: "pre-wrap",
                                maxHeight: 150,
                                overflowY: "auto",
                            }}
                        >
                            {msg}
                        </div>
                    ) : (
                        (
                            [
                                ["report_2026.pdf", "2.4 MB"],
                                ["project_notes.md", "12 KB"],
                                ["data_analysis.py", "8 KB"],
                                ["aura_config.json", "1.2 KB"],
                            ] as [string, string][]
                        ).map(([n, s]) => (
                            <div
                                key={n}
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    marginBottom: 10,
                                    paddingBottom: 9,
                                    borderBottom:
                                        "1px solid rgba(0,229,255,0.07)",
                                }}
                            >
                                <div style={{ fontSize: 12 }}>{n}</div>
                                <div style={{ fontSize: 10, opacity: 0.4 }}>
                                    {s}
                                </div>
                            </div>
                        ))
                    )}

                    <RagKnowledgeBase
                        documents={data.ragDocuments}
                        stats={data.ragStats}
                        onChanged={data.onRagChanged}
                    />
                </>
            );
        }
        default:
            return null;
    }
}

// ── Compact holographic cube ──────────────────────────────────────────────────
// Canvas-based 3D rotating cube rendered when BrowserWindow is 120×120 px.
// Purely visual — pointer events are disabled at both React and OS levels.
function CompactCube({ active }: { active: boolean }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const angleRef = useRef(Math.PI / 6);
    const timeRef = useRef(0);
    const rafRef = useRef(0);

    useEffect(() => {
        if (!active) {
            cancelAnimationFrame(rafRef.current);
            return;
        }
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d", { alpha: true })!;

        // Larger canvas to fit the glow without clipping. Center is 60,60, scale S=24.
        const W = 120,
            H = 120,
            CX = 60,
            CY = 60,
            S = 24;

        const VERTS: [number, number, number][] = [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ];
        const FACES_DEF = [
            { vi: [3, 2, 6, 7], af: 0.26, ab: 0.05 },
            { vi: [1, 2, 6, 5], af: 0.2, ab: 0.04 },
            { vi: [0, 3, 7, 4], af: 0.16, ab: 0.04 },
            { vi: [4, 5, 6, 7], af: 0.13, ab: 0.03 },
            { vi: [0, 1, 2, 3], af: 0.1, ab: 0.02 },
            { vi: [0, 1, 5, 4], af: 0.08, ab: 0.02 },
        ];
        const EDGES: [number, number][] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];

        // Rotate around Y (spin) and a gentle wobble around X for a "revolving
        // hologram" feel; constant isometric tilt keeps three faces visible.
        function project(
            v: [number, number, number],
            ry: number,
            rx: number,
        ): [number, number, number] {
            const [x, y, z] = v;
            // spin around Y
            const x1 = x * Math.cos(ry) - z * Math.sin(ry);
            const z1 = x * Math.sin(ry) + z * Math.cos(ry);
            // tilt around X (fixed iso tilt + slow wobble)
            const y2 = y * Math.cos(rx) - z1 * Math.sin(rx);
            const z2 = y * Math.sin(rx) + z1 * Math.cos(rx);
            return [CX + x1 * S, CY + y2 * S, z2 * S];
        }

        function signedArea(pts: [number, number, number][]): number {
            let a = 0;
            for (let i = 0; i < pts.length; i++) {
                const j = (i + 1) % pts.length;
                a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
            }
            return a;
        }

        function draw() {
            angleRef.current += 0.009; // slower, gentler revolving spin
            timeRef.current += 0.008;
            const ang = angleRef.current,
                t = timeRef.current;
            const tilt = 0.68 + Math.sin(t * 0.6) * 0.1; // more tilted base + slow wobble
            ctx.clearRect(0, 0, W, H);
            ctx.shadowBlur = 0;

            const pts = VERTS.map((v) => project(v, ang, tilt));

            // Sort faces back-to-front and draw
            FACES_DEF.map((f) => ({
                ...f,
                pts3: f.vi.map((i) => pts[i]) as [number, number, number][],
                avgZ: f.vi.reduce((s, i) => s + pts[i][2], 0) / 4,
            }))
                .sort((a, b) => a.avgZ - b.avgZ)
                .forEach((f) => {
                    const front = signedArea(f.pts3) < 0;
                    ctx.beginPath();
                    f.pts3.forEach(([x, y], i) =>
                        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y),
                    );
                    ctx.closePath();
                    ctx.fillStyle = `rgba(0,229,255,${front ? f.af : f.ab})`;
                    ctx.fill();
                });

            // Edges with depth brightness + energy flow animation
            EDGES.forEach(([a, b]) => {
                const [ax, ay, az] = pts[a],
                    [bx, by, bz] = pts[b];
                const depth = (az + bz) / 2;
                const norm = Math.max(
                    0,
                    Math.min(1, (depth / (S * 0.9) + 1) / 2),
                );
                const alpha = 0.5 + norm * 0.5;
                const lw = 1.4 + norm * 1.4;
                const glow = 6 + norm * 16;

                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.setLineDash([]);
                ctx.strokeStyle = `rgba(0,229,255,${alpha})`;
                ctx.lineWidth = lw;
                ctx.shadowBlur = glow;
                ctx.shadowColor = "#00e5ff";
                ctx.stroke();

                // bright white travelling energy pulse along each edge
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.setLineDash([3, 7]);
                ctx.lineDashOffset = -t * 22;
                ctx.strokeStyle = `rgba(200,248,255,${alpha * 0.85})`;
                ctx.lineWidth = lw * 0.55;
                ctx.shadowBlur = 0;
                ctx.stroke();
                ctx.setLineDash([]);
            });

            // Corner dots
            ctx.shadowColor = "#00e5ff";
            pts.forEach(([x, y, z]) => {
                const norm = Math.max(0, Math.min(1, (z / (S * 0.9) + 1) / 2));
                ctx.beginPath();
                ctx.arc(x, y, 1.3 + norm * 0.9, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0,229,255,${0.45 + norm * 0.55})`;
                ctx.shadowBlur = 8 + norm * 10;
                ctx.fill();
            });

            // Core pulse
            const pulse = 0.5 + 0.5 * Math.sin(t * 2.2);
            ctx.shadowBlur = 14 + pulse * 14;
            ctx.shadowColor = "#00e5ff";
            ctx.fillStyle = `rgba(0,229,255,${0.55 + pulse * 0.4})`;
            ctx.beginPath();
            ctx.arc(CX, CY, 2.2 + pulse * 1.4, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;

            rafRef.current = requestAnimationFrame(draw);
        }

        rafRef.current = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(rafRef.current);
    }, [active]);

    return (
        <div style={{ animation: "compFloat 3.5s ease-in-out infinite" }}>
            <canvas
                ref={canvasRef}
                width={120}
                height={120}
                style={{
                    display: "block",
                    animation: "compGlow 3s ease-in-out infinite",
                }}
            />
        </div>
    );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
    // Start in sleeping state — backend will emit SLEEPING via WakeWordGatedSession
    // the moment it boots, before the first wake word fires.
    const [astate, setAstate] = useState<AState>("sleeping");
    const [panel, setPanel] = useState<PanelId | null>(null);
    const [hovering, setHovering] = useState<PanelId | null>(null);
    const [clock, setClock] = useState("");
    const [mode, setMode] = useState<Mode>("full");

    // ── Backend connection (api/server.py over WebSocket + REST) ────────────
    const [wsConnected, setWsConnected] = useState(false);
    const [wsLatencyMs, setWsLatencyMs] = useState<number | null>(null);
    const [chatOpen, setChatOpen] = useState(false);
    const [inputText, setInputText] = useState("");
    const [lastExchange, setLastExchange] = useState<{
        who: "you" | "jarvis";
        text: string;
    } | null>(null);

    // Panel data — fetched lazily when a panel is opened (see effect below).
    // `null` means "not loaded yet / backend unavailable"; PanelContent falls
    // back to static preview content in that case.
    const [facts, setFacts] = useState<FactOut[] | null>(null);
    const [history, setHistory] = useState<HistoryTurn[] | null>(null);
    const [settingsData, setSettingsData] = useState<SettingsModel | null>(
        null,
    );
    const [alarmsMessage, setAlarmsMessage] = useState<string | null>(null);
    const [filesMessage, setFilesMessage] = useState<string | null>(null);
    const [health, setHealth] = useState<HealthInfo | null>(null);
    const [ragDocuments, setRagDocuments] = useState<RAGDocument[] | null>(null);
    const [ragStats, setRagStats] = useState<RAGStats | null>(null);
    // Bumped after an upload/delete so the FILES panel refetches the index.
    const [ragVersion, setRagVersion] = useState(0);

    const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const waveRef = useRef<SVGPathElement>(null);
    const scanRef = useRef<SVGPathElement>(null);
    const speakGRef = useRef<SVGGElement>(null);
    const orangeRef = useRef<SVGPathElement>(null);
    const rafRef = useRef<number>(0);
    const astateRef = useRef<AState>(astate);
    useEffect(() => {
        astateRef.current = astate;
    }, [astate]);

    // Clock
    useEffect(() => {
        const tick = () =>
            setClock(new Date().toLocaleTimeString("en-US", { hour12: false }));
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, []);

    // ESC closes panel / chat input
    useEffect(() => {
        const fn = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setPanel(null);
                setChatOpen(false);
            }
        };
        window.addEventListener("keydown", fn);
        return () => window.removeEventListener("keydown", fn);
    }, []);

    // SPACE wakes/sleeps the assistant (fallback for when the wake word
    // doesn't get heard) — except while chat mode is open, or while any
    // other text field is focused, where it must behave like a normal
    // space keystroke.
    useEffect(() => {
        const isTypingTarget = (target: EventTarget | null): boolean => {
            const el = target as HTMLElement | null;
            if (!el) return false;
            const tag = el.tagName;
            return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
        };
        const fn = (e: KeyboardEvent) => {
            if (e.code !== "Space" && e.key !== " ") return;
            if (chatOpen || isTypingTarget(e.target)) return;
            if (e.repeat) return; // ignore key-repeat while held down
            e.preventDefault(); // stop the page from scrolling
            wsClient.send("toggle_listen", {});
        };
        window.addEventListener("keydown", fn);
        return () => window.removeEventListener("keydown", fn);
    }, [chatOpen]);

    // ── Backend WebSocket connection ────────────────────────────────────────
    // Drives `astate` from real pipeline events (see core/events.py:
    // AssistantStateChanged) instead of only the manual debug buttons below.
    // The debug buttons still work — they call the same setAstate — useful
    // for design/demo without a running backend.
    useEffect(() => {
        wsClient.connect();

        const offOpen = wsClient.on("_open", () => setWsConnected(true));
        const offClose = wsClient.on("_close", () => setWsConnected(false));

        const offState = wsClient.on<{ state: string }>("state", (payload) => {
            if (
                payload.state === "sleeping" ||
                payload.state === "idle" ||
                payload.state === "listening" ||
                payload.state === "thinking" ||
                payload.state === "speaking"
            ) {
                setAstate(payload.state as AState);
            }
        });
        const offTranscript = wsClient.on<{ text: string; source: string }>(
            "transcript",
            (payload) => {
                setLastExchange({ who: "you", text: payload.text });
            },
        );
        const offResponse = wsClient.on<{ text: string }>(
            "response",
            (payload) => {
                setLastExchange({ who: "jarvis", text: payload.text });
            },
        );
        // Streaming assistant text — append chunks to the in-flight bubble
        // as the backend streams them (see backend PartialResponse event).
        // `done: true` finalizes; the authoritative `response` message then
        // overwrites with the exact final text.
        let streamedText = "";
        const offPartial = wsClient.on<{ text: string; done: boolean }>(
            "response_partial",
            (payload) => {
                if (payload.done) {
                    streamedText = "";
                    return;
                }
                streamedText += payload.text;
                setLastExchange({ who: "jarvis", text: streamedText });
            },
        );
        const offError = wsClient.on<{ message: string }>(
            "error",
            (payload) => {
                console.warn("[AURA] backend error:", payload.message);
            },
        );

        return () => {
            offOpen();
            offClose();
            offState();
            offTranscript();
            offPartial();
            offResponse();
            offError();
        };
    }, []);

    // Crude WS round-trip latency reading for the DEVELOPER panel — ping every
    // 10s while connected, timestamp the matching pong.
    useEffect(() => {
        if (!wsConnected) return;
        let lastPingAt = 0;
        const offPong = wsClient.on("pong", () =>
            setWsLatencyMs(
                Math.max(0, Math.round(performance.now() - lastPingAt)),
            ),
        );
        const ping = () => {
            lastPingAt = performance.now();
            wsClient.send("ping");
        };
        ping();
        const id = setInterval(ping, 10_000);
        return () => {
            clearInterval(id);
            offPong();
        };
    }, [wsConnected]);

    // ── Panel data fetch — lazily load real backend data when a panel opens ──
    useEffect(() => {
        if (!panel) return;
        let cancelled = false;

        (async () => {
            try {
                if (panel === "memory") {
                    const data = await api.getFacts();
                    if (!cancelled) setFacts(data);
                } else if (panel === "history") {
                    const data = await api.getHistory(20);
                    if (!cancelled) setHistory(data);
                } else if (panel === "settings") {
                    const data = await api.getSettings();
                    if (!cancelled) setSettingsData(data);
                } else if (panel === "automation") {
                    const data = await api.getAlarms();
                    if (!cancelled) setAlarmsMessage(data.message);
                } else if (panel === "files") {
                    // Two independent sources: the OS desktop listing and
                    // the RAG index. Settled together so one backend being
                    // down (e.g. RAG disabled) still renders the other.
                    const [files, docs, stats] = await Promise.allSettled([
                        api.listFiles("desktop"),
                        api.getRagDocuments(),
                        api.getRagStats(),
                    ]);
                    if (cancelled) return;
                    if (files.status === "fulfilled")
                        setFilesMessage(files.value.message);
                    setRagDocuments(
                        docs.status === "fulfilled" ? docs.value : [],
                    );
                    setRagStats(
                        stats.status === "fulfilled" ? stats.value : null,
                    );
                } else if (panel === "developer") {
                    const data = await api.getHealth();
                    if (!cancelled) setHealth(data);
                }
                // "tools" and "calendar" have no backend module yet — left mocked.
            } catch (err) {
                console.warn(
                    `[AURA] failed to load panel data for "${panel}":`,
                    err,
                );
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [panel, ragVersion]);

    const sendText = () => {
        const text = inputText.trim();
        if (!text) return;
        wsClient.send("user_text", { text });
        setInputText("");
    };

    const toggleTts = () => {
        const next: SettingsModel = {
            tts_enabled: !(settingsData?.tts_enabled ?? true),
            voice_mode: settingsData?.voice_mode ?? false,
            session_id: settingsData?.session_id ?? "default",
        };
        setSettingsData(next);
        api.updateSettings(next).catch((err) =>
            console.warn("[AURA] failed to save settings:", err),
        );
    };

    const toggleVoiceMode = () => {
        const next: SettingsModel = {
            tts_enabled: settingsData?.tts_enabled ?? true,
            voice_mode: !(settingsData?.voice_mode ?? false),
            session_id: settingsData?.session_id ?? "default",
        };
        setSettingsData(next);
        api.updateSettings(next).catch((err) =>
            console.warn("[AURA] failed to save settings:", err),
        );
    };

    const panelData: PanelData = {
        facts,
        history,
        settingsData,
        alarmsMessage,
        filesMessage,
        health,
        wsLatencyMs,
        onToggleTts: toggleTts,
        onToggleVoiceMode: toggleVoiceMode,
        ragDocuments,
        ragStats,
        onRagChanged: () => setRagVersion((v) => v + 1),
    };

    // IPC: main process tells us to switch mode
    // Main drives the timing — it sends enter-compact/enter-full at the right moment
    useEffect(() => {
        const offCompact = window.electronAPI?.onEnterCompact(() => {
            setPanel(null);
            setMode("compact");
        });
        const offFull = window.electronAPI?.onEnterFull(() => {
            setMode("full");
        });
        return () => {
            offCompact?.();
            offFull?.();
        };
    }, []);

    // PRIMARY mode driver — read directly off the real window size.
    // The main process resizes the BrowserWindow (900→90 / 90→900); we mirror
    // that here. This does NOT depend on IPC messages arriving, so the cube
    // reliably replaces the orb even if the enter-compact event is missed.
    useEffect(() => {
        const sync = () => {
            const compact = window.innerWidth < 300 || window.innerHeight < 300;
            setMode((prev) => {
                if (compact && prev !== "compact") {
                    setPanel(null);
                    return "compact";
                }
                if (!compact && prev !== "full") {
                    return "full";
                }
                return prev;
            });
        };
        sync();
        window.addEventListener("resize", sync);
        return () => window.removeEventListener("resize", sync);
    }, []);

    // Cleanup hover timer
    useEffect(
        () => () => {
            if (hoverTimer.current) clearTimeout(hoverTimer.current);
        },
        [],
    );

    // ── 0.5-second hover-hold ──────────────────────────────────────────────────
    const startHover = (id: PanelId) => {
        if (panel === id) return;
        if (hoverTimer.current) clearTimeout(hoverTimer.current);
        setHovering(id);
        hoverTimer.current = setTimeout(() => {
            setPanel(id);
            setHovering(null);
            hoverTimer.current = null;
        }, 500);
    };
    const endHover = () => {
        setHovering(null);
        if (hoverTimer.current) {
            clearTimeout(hoverTimer.current);
            hoverTimer.current = null;
        }
    };

    // RAF loop
    useEffect(() => {
        const t0 = performance.now();
        const frame = (now: number) => {
            const t = (now - t0) * 0.001;
            const s = astateRef.current;

            if (waveRef.current) {
                const vis = s === "listening" || s === "speaking";
                waveRef.current.style.opacity = vis ? "1" : "0";
                if (vis) {
                    const amp = s === "speaking" ? 14 : 9;
                    const N = 128;
                    const pts = Array.from({ length: N + 1 }, (_, i) => {
                        const frac = i / N,
                            a = frac * 2 * Math.PI - Math.PI / 2;
                        const noise =
                            Math.sin(frac * Math.PI * 8 + t * 3.2) * amp +
                            Math.sin(frac * Math.PI * 15 - t * 2.1) *
                                (amp * 0.5) +
                            Math.sin(frac * Math.PI * 23 + t * 4.5) *
                                (amp * 0.25);
                        const r = CORE + 8 + noise;
                        return `${i === 0 ? "M" : "L"}${(CX + r * Math.cos(a)).toFixed(1)} ${(CY + r * Math.sin(a)).toFixed(1)}`;
                    });
                    waveRef.current.setAttribute("d", pts.join(" ") + "Z");
                }
            }
            if (scanRef.current) {
                const vis = s === "thinking";
                scanRef.current.style.opacity = vis ? "0.65" : "0";
                if (vis) {
                    const a = (t * 95) % 360;
                    scanRef.current.setAttribute(
                        "d",
                        sector(CORE - 6, CORE + 20, a - 28, a + 6),
                    );
                }
            }
            if (speakGRef.current) {
                const vis = s === "speaking";
                speakGRef.current.style.opacity = vis ? "1" : "0";
                if (vis) {
                    speakGRef.current
                        .querySelectorAll("line")
                        .forEach((line, i) => {
                            const b =
                                Math.sin((i / 36) * Math.PI * 7 + t * 9) * 12 +
                                Math.sin((i / 36) * Math.PI * 13 - t * 5) * 6;
                            const r1 = CORE + 3,
                                r2 = r1 + Math.max(2, 12 + b);
                            const [x1, y1] = polar(r1, i * 10),
                                [x2, y2] = polar(r2, i * 10);
                            line.setAttribute("x1", x1.toFixed(1));
                            line.setAttribute("y1", y1.toFixed(1));
                            line.setAttribute("x2", x2.toFixed(1));
                            line.setAttribute("y2", y2.toFixed(1));
                        });
                }
            }
            if (orangeRef.current) {
                if (s === "thinking") {
                    const a = (t * 48) % 360;
                    orangeRef.current.setAttribute(
                        "d",
                        sector(122, 136, a - 18, a + 32),
                    );
                } else {
                    const end = s === "idle" ? 5 : s === "listening" ? 52 : 68;
                    orangeRef.current.setAttribute(
                        "d",
                        sector(122, 136, -18, end),
                    );
                }
            }
            rafRef.current = requestAnimationFrame(frame);
        };
        rafRef.current = requestAnimationFrame(frame);
        return () => cancelAnimationFrame(rafRef.current);
    }, []);

    const isAsleep = astate === "sleeping";
    const speedMul = isAsleep
        ? 0.08 // rings almost frozen while sleeping
        : astate === "thinking"
          ? 0.35
          : astate === "listening"
            ? 0.65
            : 1;

    const ticks = Array.from({ length: 72 }, (_, i) => {
        const long = i % 3 === 0;
        const [x1, y1] = polar(168, i * 5),
            [x2, y2] = polar(168 + (long ? 7 : 3), i * 5);
        return (
            <line
                key={i}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={CYN}
                strokeWidth={long ? 0.8 : 0.4}
                opacity={long ? 0.45 : 0.18}
            />
        );
    });

    const panelAngle = MENUS.find((m) => m.id === panel)?.angle ?? 0;
    const [clkX, clkY] = polar(235, 315);

    return (
        <div
            style={{
                position: "relative",
                width: "100vw",
                height: "100vh",
                background: "transparent",
                overflow: "hidden",
            }}
        >
            <style>{`
        @keyframes spin    { to { transform: rotate( 360deg); } }
        @keyframes unspin  { to { transform: rotate(-360deg); } }
        @keyframes breathe { 0%,100%{opacity:.5} 50%{opacity:1} }
        @keyframes breatheSlow { 0%,100%{opacity:.15} 50%{opacity:.55} }
        @keyframes flicker { 0%,95%,100%{opacity:1} 96%{opacity:.65} 97%{opacity:.9} 98%{opacity:.55} }

        @keyframes compFloat {
          0%,100% { transform: translateY(0px); }
          50%     { transform: translateY(-5px); }
        }
        @keyframes compGlow {
          0%,100% { filter: drop-shadow(0 0 8px rgba(0,229,255,0.4)) drop-shadow(0 0 20px rgba(0,229,255,0.12)); }
          50%     { filter: drop-shadow(0 0 16px rgba(0,229,255,0.75)) drop-shadow(0 0 36px rgba(0,229,255,0.25)); }
        }

        @keyframes panelReveal {
          0%   { clip-path: inset(0 0 100% 0 round 6px); }
          15%  { clip-path: inset(0 0 80%  0 round 6px); transform: translateX(-1px); }
          35%  { clip-path: inset(0 0 55%  0 round 6px); transform: translateX(2px); }
          60%  { clip-path: inset(0 0 22%  0 round 6px); transform: translateX(0); }
          100% { clip-path: inset(0 0 0%   0 round 6px); }
        }
        @keyframes sweepLine {
          0%   { top: 0%;   opacity: 1; }
          85%  { top: 95%;  opacity: .7; }
          100% { top: 100%; opacity: 0; }
        }
        .panel-reveal {
          animation: panelReveal .36s cubic-bezier(0.16,1,0.3,1) forwards;
          position: relative; overflow: hidden; border-radius: 6px;
        }
        .panel-sweep {
          position: absolute; left: 0; right: 0; height: 1px;
          background: linear-gradient(90deg, transparent, #00e5ff 40%, #fff 50%, #00e5ff 60%, transparent);
          box-shadow: 0 0 8px #00e5ff;
          animation: sweepLine .36s ease-out forwards;
          z-index: 10; pointer-events: none;
        }
        .drag-region { -webkit-app-region: drag; }
        .no-drag     { -webkit-app-region: no-drag; }
        .aura-ring   { transform-box: fill-box; transform-origin: center; }
      `}</style>

            {/* ── COMPACT MODE
            The BrowserWindow is now 120×120 px sitting at the screen's bottom-right.
            This layer fills 100vw/100vh (= 120×120) and shows the holographic cube.
            In full mode it is invisible (opacity 0, pointer-events none).
            OS-level setIgnoreMouseEvents handles click-through; pointerEvents none
            here is belt-and-suspenders.
      ─────────────────────────────────────────────────────────────────────────── */}
            <motion.div
                title="Click to restore AURA"
                onClick={() => {
                    if (mode === "compact")
                        window.electronAPI?.requestRestore();
                }}
                style={{
                    position: "absolute",
                    inset: 0,
                    zIndex: 200,
                    // Clickable ONLY in compact mode — clicking the cube restores the orb.
                    pointerEvents: mode === "compact" ? "auto" : "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: mode === "compact" ? "pointer" : "default",
                }}
                animate={{
                    opacity: mode === "compact" ? 1 : 0,
                    scale: mode === "compact" ? 1 : 0.4,
                }}
                transition={
                    mode === "compact"
                        ? // Cube materialises as the window is still collapsing, so you see a
                          // morph rather than a shrink-then-pop.
                          {
                              opacity: {
                                  delay: 0.08,
                                  duration: 0.28,
                                  ease: "easeOut",
                              },
                              scale: {
                                  type: "spring",
                                  stiffness: 260,
                                  damping: 20,
                                  delay: 0.08,
                              },
                          }
                        : { duration: 0.1 }
                }
            >
                <CompactCube active={mode === "compact"} />
            </motion.div>

            {/* ── FULL MODE
            The BrowserWindow is 900×900 px centered on screen.
            All orb rendering, panels, and UI live here.
            Fades out when compact mode starts; springs back in when restored.
      ─────────────────────────────────────────────────────────────────────────── */}
            <motion.div
                style={{
                    position: "absolute",
                    inset: 0,
                    pointerEvents: mode === "full" ? "auto" : "none",
                }}
                animate={
                    mode === "full"
                        ? { opacity: 1, scale: 1 }
                        : { opacity: 0, scale: 0.82 }
                }
                transition={
                    mode === "full"
                        ? // Spring expansion — orb grows in sync with the window expanding
                          {
                              type: "spring",
                              stiffness: 160,
                              damping: 22,
                              opacity: { duration: 0.3 },
                          }
                        : { duration: 0.22, ease: "easeIn" }
                }
            >
                {/* Drag handle — transparent circle over the orb core */}
                <div
                    className="drag-region"
                    style={{
                        position: "absolute",
                        left: "50%",
                        top: "50%",
                        width: "160px",
                        height: "160px",
                        transform: "translate(-50%, -50%)",
                        borderRadius: "50%",
                        zIndex: 5,
                        cursor: "move",
                    }}
                />

                <svg
                    viewBox={`0 0 ${VW} ${VH}`}
                    width="100%"
                    height="100%"
                    className="no-drag"
                    style={{
                        display: "block",
                        position: "absolute",
                        inset: 0,
                        // Smoothly dim everything when sleeping; orb looks powered down.
                        filter: isAsleep
                            ? "brightness(0.22) saturate(0.4)"
                            : undefined,
                        transition: "filter 2s ease",
                    }}
                >
                    <defs>
                        <radialGradient id="core-g" cx="50%" cy="50%" r="50%">
                            <stop
                                offset="0%"
                                stopColor="#00e5ff"
                                stopOpacity=".22"
                            />
                            <stop
                                offset="45%"
                                stopColor="#003850"
                                stopOpacity=".3"
                            />
                            <stop
                                offset="100%"
                                stopColor="#000d18"
                                stopOpacity=".98"
                            />
                        </radialGradient>
                        <radialGradient id="bg-g" cx="50%" cy="50%" r="50%">
                            <stop
                                offset="0%"
                                stopColor="#00e5ff"
                                stopOpacity=".10"
                            />
                            <stop
                                offset="55%"
                                stopColor="#00e5ff"
                                stopOpacity=".03"
                            />
                            <stop
                                offset="100%"
                                stopColor="#00e5ff"
                                stopOpacity="0"
                            />
                        </radialGradient>
                        <filter
                            id="glow3"
                            x="-40%"
                            y="-40%"
                            width="180%"
                            height="180%"
                        >
                            <feGaussianBlur stdDeviation="3" result="b" />
                            <feMerge>
                                <feMergeNode in="b" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                        <filter
                            id="blur5"
                            x="-80%"
                            y="-80%"
                            width="260%"
                            height="260%"
                        >
                            <feGaussianBlur stdDeviation="5" />
                        </filter>
                    </defs>

                    <ellipse
                        cx={CX}
                        cy={CY}
                        rx={220}
                        ry={210}
                        fill="url(#bg-g)"
                    />

                    {RINGS.map((ring, i) => (
                        <circle
                            key={i}
                            cx={CX}
                            cy={CY}
                            r={ring.r}
                            fill="none"
                            stroke={i % 2 === 0 ? CYN : CYN2}
                            strokeWidth={ring.w}
                            strokeDasharray={ring.d}
                            opacity={ring.o}
                            className="aura-ring"
                            style={{
                                animationName: ring.cw ? "spin" : "unspin",
                                animationDuration: `${ring.s * speedMul}s`,
                                animationTimingFunction: "linear",
                                animationIterationCount: "infinite",
                            }}
                        />
                    ))}
                    {ticks}

                    <path
                        ref={orangeRef}
                        d={sector(122, 136, -18, 5)}
                        fill={ORG}
                        opacity=".78"
                        filter="url(#glow3)"
                    />

                    {(astate === "listening" || astate === "speaking") &&
                        [0, 0.7, 1.4].map((d) => (
                            <circle
                                key={d}
                                cx={CX}
                                cy={CY}
                                r={CORE}
                                fill="none"
                                stroke={CYN}
                                strokeWidth="1.5"
                                opacity="0"
                            >
                                <animate
                                    attributeName="r"
                                    from={CORE}
                                    to={195}
                                    dur="2.2s"
                                    begin={`${d}s`}
                                    repeatCount="indefinite"
                                />
                                <animate
                                    attributeName="opacity"
                                    from=".55"
                                    to="0"
                                    dur="2.2s"
                                    begin={`${d}s`}
                                    repeatCount="indefinite"
                                />
                            </circle>
                        ))}

                    <circle
                        cx={CX}
                        cy={CY}
                        r={CORE + 18}
                        fill={CYN}
                        opacity=".04"
                        style={{
                            animation: "breathe 3.5s ease-in-out infinite",
                        }}
                    />
                    <circle
                        cx={CX}
                        cy={CY}
                        r={CORE + 9}
                        fill={CYN}
                        opacity=".07"
                        style={{
                            animation: "breathe 3.5s ease-in-out infinite",
                            animationDelay: ".7s",
                        }}
                    />

                    <path
                        ref={waveRef}
                        fill="none"
                        stroke={CYN}
                        strokeWidth="1.6"
                        style={{
                            opacity: 0,
                            transition: "opacity .4s",
                            filter: "url(#glow3)",
                        }}
                    />
                    <path
                        ref={scanRef}
                        fill={CYN}
                        opacity="0"
                        style={{
                            transition: "opacity .3s",
                            filter: "url(#blur5)",
                        }}
                    />
                    <g
                        ref={speakGRef}
                        style={{ opacity: 0, transition: "opacity .3s" }}
                    >
                        {Array.from({ length: 36 }, (_, i) => (
                            <line
                                key={i}
                                stroke={CYN}
                                strokeWidth="1.6"
                                opacity=".6"
                            />
                        ))}
                    </g>

                    <circle
                        cx={CX}
                        cy={CY}
                        r={CORE}
                        fill="url(#core-g)"
                        stroke={CYN}
                        strokeWidth="1.4"
                        filter="url(#glow3)"
                        style={{
                            animation: "breathe 4.2s ease-in-out infinite",
                        }}
                    />
                    <circle
                        cx={CX}
                        cy={CY}
                        r={CORE - 12}
                        fill="none"
                        stroke={CYN}
                        strokeWidth=".4"
                        opacity=".2"
                    />
                    <circle
                        cx={CX}
                        cy={CY}
                        r={CORE - 24}
                        fill="none"
                        stroke={CYN}
                        strokeWidth=".3"
                        opacity=".12"
                    />
                    <line
                        x1={CX - 11}
                        y1={CY}
                        x2={CX + 11}
                        y2={CY}
                        stroke={CYN}
                        strokeWidth=".4"
                        opacity=".2"
                    />
                    <line
                        x1={CX}
                        y1={CY - 11}
                        x2={CX}
                        y2={CY + 11}
                        stroke={CYN}
                        strokeWidth=".4"
                        opacity=".2"
                    />
                    {[45, 135, 225, 315].map((a) => {
                        const [x1, y1] = polar(CORE - 3, a),
                            [x2, y2] = polar(CORE - 9, a);
                        return (
                            <line
                                key={a}
                                x1={x1}
                                y1={y1}
                                x2={x2}
                                y2={y2}
                                stroke={CYN}
                                strokeWidth=".9"
                                opacity=".32"
                            />
                        );
                    })}

                    <text
                        x={CX}
                        y={CY - 12}
                        textAnchor="middle"
                        fill="white"
                        fontSize="17"
                        fontFamily="Orbitron,monospace"
                        fontWeight="700"
                        letterSpacing="6"
                        style={{
                            animation: "flicker 14s ease-in-out infinite",
                        }}
                    >
                        JARVIS
                    </text>
                    <text
                        x={CX}
                        y={CY + 4}
                        textAnchor="middle"
                        fill={CYN}
                        fontSize="6.5"
                        fontFamily="'Share Tech Mono',monospace"
                        letterSpacing="2"
                        opacity=".4"
                    >
                        v2.6.3
                    </text>
                    <text
                        x={CX}
                        y={CY + 20}
                        textAnchor="middle"
                        fill={isAsleep ? "rgba(0,229,255,.55)" : CYN}
                        fontSize={isAsleep ? "6.5" : "8.5"}
                        fontFamily="'Share Tech Mono',monospace"
                        letterSpacing={isAsleep ? "1.5" : "2"}
                        style={{
                            animation: isAsleep
                                ? "breatheSlow 5s ease-in-out infinite"
                                : "breathe 2s ease-in-out infinite",
                        }}
                    >
                        {STATE_LBL[astate]}
                    </text>

                    {/* Clock */}
                    <text
                        x={clkX}
                        y={clkY}
                        textAnchor="middle"
                        fill={CYN}
                        fontSize="7.5"
                        fontFamily="'Share Tech Mono',monospace"
                        opacity=".38"
                        letterSpacing="1"
                    >
                        {clock}
                    </text>

                    {/* Radial menu */}
                    {MENUS.map((m) => {
                        const isOpen = panel === m.id;
                        const isHover = hovering === m.id;
                        const R1 = 172,
                            R2 = 208,
                            SPAN = 30;
                        const [lx, ly] = polar(220, m.angle);
                        return (
                            <g
                                key={m.id}
                                className="no-drag"
                                style={{ cursor: "crosshair" }}
                                onMouseEnter={() => startHover(m.id)}
                                onMouseLeave={() => endHover()}
                            >
                                <path
                                    d={sector(
                                        R1,
                                        R2,
                                        m.angle - SPAN / 2,
                                        m.angle + SPAN / 2,
                                    )}
                                    fill="transparent"
                                    stroke="none"
                                />
                                <text
                                    x={lx}
                                    y={ly + 3}
                                    textAnchor="middle"
                                    fill={
                                        isOpen
                                            ? CYN
                                            : isHover
                                              ? "white"
                                              : `rgba(0,229,255,0.32)`
                                    }
                                    fontSize="6.5"
                                    fontFamily="'Share Tech Mono',monospace"
                                    letterSpacing="1.5"
                                    filter={
                                        isOpen || isHover
                                            ? "url(#glow3)"
                                            : undefined
                                    }
                                    style={{
                                        userSelect: "none",
                                        transition: "all .2s ease",
                                    }}
                                >
                                    {m.label}
                                </text>

                                {isHover && !isOpen && (
                                    <motion.path
                                        key={`p-${m.id}`}
                                        d={arcOnly(
                                            212,
                                            m.angle - 15,
                                            m.angle + 15,
                                        )}
                                        fill="none"
                                        stroke={CYN}
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        filter="url(#glow3)"
                                        initial={{ pathLength: 0, opacity: 0 }}
                                        animate={{ pathLength: 1, opacity: 1 }}
                                        transition={{
                                            pathLength: {
                                                duration: 0.5,
                                                ease: "linear",
                                            },
                                            opacity: { duration: 0.08 },
                                        }}
                                    />
                                )}

                                {isOpen &&
                                    (() => {
                                        const [dx, dy] = polar(213, m.angle);
                                        return (
                                            <circle
                                                cx={dx}
                                                cy={dy}
                                                r="2"
                                                fill={CYN}
                                                filter="url(#glow3)"
                                                opacity=".9"
                                            />
                                        );
                                    })()}
                            </g>
                        );
                    })}

                    {panel &&
                        (() => {
                            const m = MENUS.find((x) => x.id === panel)!;
                            return (
                                <path
                                    d={sector(
                                        200,
                                        240,
                                        m.angle - 16,
                                        m.angle + 16,
                                    )}
                                    fill={CYN}
                                    opacity=".07"
                                    filter="url(#blur5)"
                                />
                            );
                        })()}

                    {/* Chat toggle — centered below the orb */}
                    <g
                        className="no-drag"
                        transform={`translate(${CX - 34},${VH - 30})`}
                    >
                        <g
                            style={{ cursor: "pointer" }}
                            onClick={() => setChatOpen((v) => !v)}
                        >
                            <rect
                                x={0}
                                y={0}
                                width={68}
                                height={18}
                                rx={3}
                                fill={
                                    chatOpen
                                        ? "rgba(0,255,136,.16)"
                                        : "rgba(0,229,255,.03)"
                                }
                                stroke={chatOpen ? GRN : "rgba(0,229,255,.14)"}
                                strokeWidth={chatOpen ? 1.1 : 0.4}
                                filter={chatOpen ? "url(#glow3)" : undefined}
                            />
                            <circle
                                cx={8}
                                cy={9}
                                r={2.5}
                                fill={wsConnected ? GRN : RED}
                            />
                            <text
                                x={40}
                                y={12}
                                textAnchor="middle"
                                fill={chatOpen ? GRN : "rgba(0,229,255,.26)"}
                                fontSize="6"
                                fontFamily="'Share Tech Mono',monospace"
                                letterSpacing="0.5"
                            >
                                CHAT
                            </text>
                        </g>
                    </g>
                </svg>

                {/* Panel overlays */}
                <AnimatePresence mode="wait">
                    {panel && (
                        <motion.div
                            key={panel}
                            custom={panelAngle}
                            variants={PANEL_VARIANTS}
                            initial="enter"
                            animate="center"
                            exit="exit"
                            className="no-drag"
                            style={{
                                position: "absolute",
                                ...PANEL_POS[panel],
                                width: "260px",
                                maxHeight: "240px",
                                zIndex: 20,
                            }}
                        >
                            <div
                                className="panel-reveal"
                                style={{
                                    background: "rgba(0,7,14,0.94)",
                                    border: "1px solid rgba(0,229,255,0.30)",
                                    borderRadius: "8px",
                                    padding: "18px 18px 16px",
                                    backdropFilter: "blur(18px)",
                                    WebkitBackdropFilter: "blur(18px)",
                                    color: CYN,
                                    fontFamily: "'Share Tech Mono',monospace",
                                    fontSize: "12px",
                                    boxShadow:
                                        "0 0 40px rgba(0,229,255,.08), inset 0 0 0 1px rgba(0,229,255,.04)",
                                    overflowY: "auto",
                                    maxHeight: "240px",
                                    scrollbarWidth: "none" as const,
                                }}
                            >
                                <div className="panel-sweep" />
                                <div
                                    style={{
                                        position: "absolute",
                                        top: 0,
                                        left: "10%",
                                        right: "10%",
                                        height: "1px",
                                        background: `linear-gradient(90deg,transparent,${CYN},transparent)`,
                                        opacity: 0.5,
                                    }}
                                />
                                {(
                                    [
                                        [0, 0, 1, 1],
                                        [1, 0, -1, 1],
                                        [0, 1, 1, -1],
                                        [1, 1, -1, -1],
                                    ] as number[][]
                                ).map(([rx, ry, sx, sy], i) => (
                                    <div
                                        key={i}
                                        style={{
                                            position: "absolute",
                                            [rx ? "right" : "left"]: 7,
                                            [ry ? "bottom" : "top"]: 7,
                                            width: 12,
                                            height: 12,
                                            borderTop:
                                                sy === 1
                                                    ? "1px solid rgba(0,229,255,.4)"
                                                    : "none",
                                            borderBottom:
                                                sy === -1
                                                    ? "1px solid rgba(0,229,255,.4)"
                                                    : "none",
                                            borderLeft:
                                                sx === 1
                                                    ? "1px solid rgba(0,229,255,.4)"
                                                    : "none",
                                            borderRight:
                                                sx === -1
                                                    ? "1px solid rgba(0,229,255,.4)"
                                                    : "none",
                                        }}
                                    />
                                ))}
                                <PanelContent id={panel} data={panelData} />
                                <button
                                    onClick={() => setPanel(null)}
                                    style={{
                                        position: "absolute",
                                        top: 10,
                                        right: 10,
                                        background: "transparent",
                                        border: "1px solid rgba(0,229,255,.28)",
                                        color: "rgba(0,229,255,.6)",
                                        fontSize: 16,
                                        lineHeight: 1,
                                        width: 22,
                                        height: 22,
                                        cursor: "pointer",
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        borderRadius: 3,
                                        fontFamily: "monospace",
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Chat input — toggled by the CHAT pill next to the state debug
                    controls. Sends over the WebSocket to api/server.py, which
                    injects it onto the bus as a TextInputReceived event. */}
                <AnimatePresence>
                    {chatOpen && (
                        <motion.div
                            key="chat-input"
                            className="no-drag"
                            initial={{ opacity: 0, y: 10, scale: 0.96 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 10, scale: 0.96 }}
                            transition={{ duration: 0.18 }}
                            style={{
                                position: "absolute",
                                left: "50%",
                                bottom: "72px",
                                transform: "translateX(-50%)",
                                width: "320px",
                                zIndex: 30,
                            }}
                        >
                            <form
                                onSubmit={(e) => {
                                    e.preventDefault();
                                    sendText();
                                }}
                                style={{
                                    display: "flex",
                                    gap: "6px",
                                    background: "rgba(0,7,14,0.94)",
                                    border: "1px solid rgba(0,229,255,0.30)",
                                    borderRadius: "8px",
                                    padding: "8px 10px",
                                    backdropFilter: "blur(18px)",
                                    WebkitBackdropFilter: "blur(18px)",
                                    boxShadow: "0 0 40px rgba(0,229,255,.08)",
                                }}
                            >
                                <input
                                    autoFocus
                                    value={inputText}
                                    onChange={(e) =>
                                        setInputText(e.target.value)
                                    }
                                    placeholder={
                                        wsConnected
                                            ? "Ask JARVIS…"
                                            : "Backend offline…"
                                    }
                                    style={{
                                        flex: 1,
                                        background: "transparent",
                                        border: "none",
                                        outline: "none",
                                        color: CYN,
                                        fontFamily:
                                            "'Share Tech Mono',monospace",
                                        fontSize: "12px",
                                    }}
                                />
                                <button
                                    type="submit"
                                    disabled={!wsConnected}
                                    style={{
                                        background: "transparent",
                                        border: "1px solid rgba(0,229,255,.4)",
                                        borderRadius: "4px",
                                        color: CYN,
                                        fontFamily:
                                            "'Share Tech Mono',monospace",
                                        fontSize: "10px",
                                        padding: "4px 10px",
                                        cursor: wsConnected
                                            ? "pointer"
                                            : "not-allowed",
                                        opacity: wsConnected ? 1 : 0.4,
                                    }}
                                >
                                    SEND
                                </button>
                            </form>
                            {lastExchange && (
                                <div
                                    style={{
                                        marginTop: "6px",
                                        fontSize: "10px",
                                        fontFamily:
                                            "'Share Tech Mono',monospace",
                                        color:
                                            lastExchange.who === "you"
                                                ? "rgba(255,255,255,.6)"
                                                : CYN,
                                        background: "rgba(0,7,14,0.85)",
                                        border: "1px solid rgba(0,229,255,0.15)",
                                        borderRadius: "6px",
                                        padding: "6px 8px",
                                        maxHeight: "80px",
                                        overflowY: "auto",
                                    }}
                                >
                                    <span style={{ opacity: 0.6 }}>
                                        {lastExchange.who === "you"
                                            ? "YOU: "
                                            : "JARVIS: "}
                                    </span>
                                    {lastExchange.text}
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}
