import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";

declare global {
  interface Window {
    electronAPI?: {
      onEnterCompact: (cb: () => void) => (() => void);
      onEnterFull:    (cb: () => void) => (() => void);
      requestRestore: () => void;
    };
  }
}

type AState  = "idle" | "listening" | "thinking" | "speaking";
type PanelId = "settings" | "history" | "memory" | "tools" | "automation" | "calendar" | "developer" | "files";
type Mode    = "full" | "compact";

// ── Canvas ────────────────────────────────────────────────────────────────────
const VW = 520, VH = 520, CX = 260, CY = 260, CORE = 80;
const CYN = "#00e5ff", CYN2 = "#00b8d4";
const ORG = "#ff9800", GRN = "#00ff88", RED = "#ff4444";

// ── Geometry ──────────────────────────────────────────────────────────────────
function polar(r: number, deg: number): [number, number] {
  const a = (deg - 90) * (Math.PI / 180);
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}
function sector(r1: number, r2: number, a1: number, a2: number): string {
  const [x1,y1]=polar(r1,a1),[x2,y2]=polar(r1,a2);
  const [x3,y3]=polar(r2,a2),[x4,y4]=polar(r2,a1);
  const lg=a2-a1>180?1:0;
  return `M${x1} ${y1} A${r1} ${r1} 0 ${lg} 1 ${x2} ${y2} L${x3} ${y3} A${r2} ${r2} 0 ${lg} 0 ${x4} ${y4}Z`;
}
function arcOnly(r: number, a1: number, a2: number): string {
  const [x1,y1]=polar(r,a1),[x2,y2]=polar(r,a2);
  const lg=a2-a1>180?1:0;
  return `M${x1} ${y1} A${r} ${r} 0 ${lg} 1 ${x2} ${y2}`;
}

// ── Config ────────────────────────────────────────────────────────────────────
const RINGS = [
  { r: 92,  w:1.5, d:"3 6",  o:.70, s:20, cw:true  },
  { r:103,  w:1.0, d:"1 8",  o:.40, s:28, cw:false },
  { r:115,  w:1.5, d:"8 3",  o:.55, s:40, cw:true  },
  { r:127,  w:0.8, d:"2 5",  o:.30, s:50, cw:false },
  { r:140,  w:1.2, d:"6 5",  o:.45, s:30, cw:true  },
  { r:152,  w:0.6, d:"1 4",  o:.25, s:70, cw:false },
  { r:165,  w:1.0, d:"4 3",  o:.50, s:55, cw:true  },
];

const MENUS: { id:PanelId; label:string; angle:number }[] = [
  { id:"developer",  label:"DEV",      angle:0   },
  { id:"settings",   label:"SETTINGS", angle:45  },
  { id:"tools",      label:"TOOLS",    angle:90  },
  { id:"history",    label:"HISTORY",  angle:135 },
  { id:"memory",     label:"MEMORY",   angle:180 },
  { id:"files",      label:"FILES",    angle:225 },
  { id:"calendar",   label:"CALENDAR", angle:270 },
  { id:"automation", label:"AUTO",     angle:315 },
];

const STATE_LBL: Record<AState,string> = {
  idle:"IDLE", listening:"LISTENING...", thinking:"THINKING...", speaking:"SPEAKING...",
};

const PANEL_POS: Record<PanelId,React.CSSProperties> = {
  developer:  { top:"14px",   left:"50%",  transform:"translateX(-50%)" },
  settings:   { top:"14px",   right:"14px" },
  tools:      { top:"50%",    right:"14px", transform:"translateY(-50%)" },
  history:    { bottom:"14px",right:"14px" },
  memory:     { bottom:"14px",left:"50%",   transform:"translateX(-50%)" },
  files:      { bottom:"14px",left:"14px"  },
  calendar:   { top:"50%",    left:"14px",  transform:"translateY(-50%)" },
  automation: { top:"14px",   left:"14px"  },
};

const PANEL_VARIANTS = {
  enter: (angle: number) => ({
    opacity: 0,
    x:  Math.sin(angle * Math.PI / 180) * 30,
    y: -Math.cos(angle * Math.PI / 180) * 30,
    scale: 0.84, filter: "blur(8px)",
  }),
  center: {
    opacity: 1, x: 0, y: 0, scale: 1, filter: "blur(0px)",
    transition: { type: "spring" as const, stiffness: 380, damping: 28 },
  },
  exit: (angle: number) => ({
    opacity: 0,
    x: -Math.sin(angle * Math.PI / 180) * 20,
    y:  Math.cos(angle * Math.PI / 180) * 20,
    scale: 0.91, filter: "blur(6px)",
    transition: { duration: 0.18, ease: "easeIn" as const },
  }),
};

// ── Panel content ─────────────────────────────────────────────────────────────
function PanelContent({ id }: { id: PanelId }) {
  const row: React.CSSProperties = {
    display:"flex", justifyContent:"space-between",
    marginBottom:8, paddingBottom:7,
    borderBottom:"1px solid rgba(0,229,255,0.08)", fontSize:12,
  };
  const dim: React.CSSProperties = { opacity:.5 };
  const val: React.CSSProperties = { color:GRN };

  switch (id) {
    case "settings": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>SETTINGS</h3>
      {([["VOICE MODE","ENABLED"],["LANGUAGE","EN-US"],["LLM PROVIDER","GPT-4o"],["MEMORY","ACTIVE"],["THEME","DARK"]] as [string,string][]).map(([k,v]) => (
        <div key={k} style={row}><span style={dim}>{k}</span><span style={val}>{v}</span></div>
      ))}
    </>);
    case "history": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>HISTORY</h3>
      {([["Quantum computing","2h ago"],["Sales analysis","5h ago"],["Code debugging","Yesterday"],["Trip planning","2d ago"]] as [string,string][]).map(([t,d]) => (
        <div key={t} style={{borderLeft:"2px solid rgba(0,229,255,0.3)",paddingLeft:8,marginBottom:10}}>
          <div style={{fontSize:12}}>{t}</div><div style={{fontSize:10,opacity:.4,marginTop:2}}>{d}</div>
        </div>
      ))}
    </>);
    case "memory": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>MEMORY</h3>
      {["Dark theme preferred","Language: English","Technology sector","Uses VS Code","Based in New York"].map(m => (
        <div key={m} style={{display:"flex",gap:7,marginBottom:9,fontSize:12,alignItems:"flex-start"}}><span style={{color:CYN2,flexShrink:0}}>▸</span><span>{m}</span></div>
      ))}
    </>);
    case "tools": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>TOOLS</h3>
      {([["Web Search",true],["Code Executor",true],["File Manager",true],["Browser Ctrl",false],["Vision API",true]] as [string,boolean][]).map(([n,a]) => (
        <div key={String(n)} style={{...row,alignItems:"center"}}><span style={dim}>{String(n)}</span>
          <span style={{fontSize:10,color:a?GRN:RED,border:`1px solid ${a?"rgba(0,255,136,0.3)":"rgba(255,68,68,0.3)"}`,padding:"2px 7px",borderRadius:2}}>{a?"ON":"OFF"}</span>
        </div>
      ))}
    </>);
    case "calendar": return (<>
      <h3 style={{margin:"0 0 6px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>CALENDAR</h3>
      <div style={{fontSize:10,opacity:.38,marginBottom:12}}>{new Date().toLocaleDateString("en-US",{month:"short",day:"numeric"})}</div>
      {([["09:30","Quantum Review",CYN],["11:00","Standup",GRN],["14:00","Demo",ORG],["16:30","Call",CYN]] as [string,string,string][]).map(([t,e,c]) => (
        <div key={e} style={{display:"flex",gap:8,marginBottom:9,alignItems:"center"}}><span style={{color:c,minWidth:38,fontSize:11,fontWeight:700}}>{t}</span><span style={{fontSize:12,borderLeft:`1px solid ${c}40`,paddingLeft:7}}>{e}</span></div>
      ))}
    </>);
    case "automation": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>AUTOMATION</h3>
      {([["Daily Brief","08:00",true],["Email Digest","18:00",true],["News Summary","07:30",false],["Health Check","12:00",true]] as [string,string,boolean][]).map(([n,s,a]) => (
        <div key={String(n)} style={{marginBottom:10,paddingBottom:9,borderBottom:"1px solid rgba(0,229,255,0.08)"}}>
          <div style={{display:"flex",justifyContent:"space-between",fontSize:12,alignItems:"center"}}><span>{String(n)}</span><span style={{fontSize:10,color:a?GRN:ORG}}>{a?"ON":"OFF"}</span></div>
          <div style={{fontSize:10,opacity:.35,marginTop:2}}>{String(s)}</div>
        </div>
      ))}
    </>);
    case "developer": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>DEVELOPER</h3>
      {([["STATUS","CONNECTED",GRN],["MODEL","GPT-4o",CYN],["LATENCY","234ms",CYN],["STREAM","ENABLED",GRN],["VERSION","v2.6.3",CYN2]] as [string,string,string][]).map(([k,v,c]) => (
        <div key={k} style={row}><span style={dim}>{k}</span><span style={{color:c}}>{v}</span></div>
      ))}
    </>);
    case "files": return (<>
      <h3 style={{margin:"0 0 14px",fontSize:11,letterSpacing:3,color:"white",fontWeight:700}}>FILES</h3>
      {([["report_2026.pdf","2.4 MB"],["project_notes.md","12 KB"],["data_analysis.py","8 KB"],["aura_config.json","1.2 KB"]] as [string,string][]).map(([n,s]) => (
        <div key={n} style={{display:"flex",justifyContent:"space-between",marginBottom:10,paddingBottom:9,borderBottom:"1px solid rgba(0,229,255,0.07)"}}>
          <div style={{fontSize:12}}>{n}</div><div style={{fontSize:10,opacity:.4}}>{s}</div>
        </div>
      ))}
    </>);
    default: return null;
  }
}

// ── Compact holographic cube ──────────────────────────────────────────────────
// Canvas-based 3D rotating cube rendered when BrowserWindow is 120×120 px.
// Purely visual — pointer events are disabled at both React and OS levels.
function CompactCube({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const angleRef  = useRef(Math.PI / 6)
  const timeRef   = useRef(0)
  const rafRef    = useRef(0)

  useEffect(() => {
    if (!active) { cancelAnimationFrame(rafRef.current); return }
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })!

    // Larger canvas to fit the glow without clipping. Center is 60,60, scale S=24.
    const W = 120, H = 120, CX = 60, CY = 60, S = 24

    const VERTS: [number,number,number][] = [
      [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
      [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ]
    const FACES_DEF = [
      {vi:[3,2,6,7],af:0.26,ab:0.05},
      {vi:[1,2,6,5],af:0.20,ab:0.04},
      {vi:[0,3,7,4],af:0.16,ab:0.04},
      {vi:[4,5,6,7],af:0.13,ab:0.03},
      {vi:[0,1,2,3],af:0.10,ab:0.02},
      {vi:[0,1,5,4],af:0.08,ab:0.02},
    ]
    const EDGES: [number,number][] = [
      [0,1],[1,2],[2,3],[3,0],
      [4,5],[5,6],[6,7],[7,4],
      [0,4],[1,5],[2,6],[3,7],
    ]

    // Rotate around Y (spin) and a gentle wobble around X for a "revolving
    // hologram" feel; constant isometric tilt keeps three faces visible.
    function project(v:[number,number,number], ry:number, rx:number):[number,number,number] {
      const [x,y,z] = v
      // spin around Y
      const x1 = x*Math.cos(ry) - z*Math.sin(ry)
      const z1 = x*Math.sin(ry) + z*Math.cos(ry)
      // tilt around X (fixed iso tilt + slow wobble)
      const y2 = y*Math.cos(rx) - z1*Math.sin(rx)
      const z2 = y*Math.sin(rx) + z1*Math.cos(rx)
      return [CX + x1*S, CY + y2*S, z2*S]
    }

    function signedArea(pts:[number,number,number][]):number {
      let a = 0
      for (let i=0;i<pts.length;i++) {
        const j=(i+1)%pts.length
        a += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
      }
      return a
    }

    function draw() {
      angleRef.current += 0.009            // slower, gentler revolving spin
      timeRef.current  += 0.008
      const ang = angleRef.current, t = timeRef.current
      const tilt = 0.68 + Math.sin(t * 0.6) * 0.10   // more tilted base + slow wobble
      ctx.clearRect(0,0,W,H)
      ctx.shadowBlur = 0

      const pts = VERTS.map(v => project(v, ang, tilt))

      // Sort faces back-to-front and draw
      FACES_DEF
        .map(f => ({...f, pts3: f.vi.map(i=>pts[i]) as [number,number,number][], avgZ: f.vi.reduce((s,i)=>s+pts[i][2],0)/4}))
        .sort((a,b)=>a.avgZ-b.avgZ)
        .forEach(f => {
          const front = signedArea(f.pts3) < 0
          ctx.beginPath()
          f.pts3.forEach(([x,y],i) => i===0?ctx.moveTo(x,y):ctx.lineTo(x,y))
          ctx.closePath()
          ctx.fillStyle = `rgba(0,229,255,${front?f.af:f.ab})`
          ctx.fill()
        })

      // Edges with depth brightness + energy flow animation
      EDGES.forEach(([a,b]) => {
        const [ax,ay,az]=pts[a],[bx,by,bz]=pts[b]
        const depth=(az+bz)/2
        const norm=Math.max(0,Math.min(1,(depth/(S*0.9)+1)/2))
        const alpha=0.5+norm*0.5
        const lw=1.4+norm*1.4
        const glow=6+norm*16

        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by)
        ctx.setLineDash([])
        ctx.strokeStyle=`rgba(0,229,255,${alpha})`
        ctx.lineWidth=lw
        ctx.shadowBlur=glow; ctx.shadowColor='#00e5ff'
        ctx.stroke()

        // bright white travelling energy pulse along each edge
        ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by)
        ctx.setLineDash([3,7])
        ctx.lineDashOffset=-t*22
        ctx.strokeStyle=`rgba(200,248,255,${alpha*0.85})`
        ctx.lineWidth=lw*0.55
        ctx.shadowBlur=0
        ctx.stroke()
        ctx.setLineDash([])
      })

      // Corner dots
      ctx.shadowColor='#00e5ff'
      pts.forEach(([x,y,z]) => {
        const norm=Math.max(0,Math.min(1,(z/(S*0.9)+1)/2))
        ctx.beginPath()
        ctx.arc(x,y,1.3+norm*0.9,0,Math.PI*2)
        ctx.fillStyle=`rgba(0,229,255,${0.45+norm*0.55})`
        ctx.shadowBlur=8+norm*10
        ctx.fill()
      })

      // Core pulse
      const pulse=0.5+0.5*Math.sin(t*2.2)
      ctx.shadowBlur=14+pulse*14; ctx.shadowColor='#00e5ff'
      ctx.fillStyle=`rgba(0,229,255,${0.55+pulse*0.4})`
      ctx.beginPath(); ctx.arc(CX,CY,2.2+pulse*1.4,0,Math.PI*2); ctx.fill()
      ctx.shadowBlur=0

      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [active])

  return (
    <div style={{animation:'compFloat 3.5s ease-in-out infinite'}}>
      <canvas
        ref={canvasRef} width={120} height={120}
        style={{display:'block', animation:'compGlow 3s ease-in-out infinite'}}
      />
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [astate,   setAstate]   = useState<AState>("idle");
  const [panel,    setPanel]    = useState<PanelId | null>(null);
  const [hovering, setHovering] = useState<PanelId | null>(null);
  const [clock,    setClock]    = useState("");
  const [mode,     setMode]     = useState<Mode>("full");

  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const waveRef    = useRef<SVGPathElement>(null);
  const scanRef    = useRef<SVGPathElement>(null);
  const speakGRef  = useRef<SVGGElement>(null);
  const orangeRef  = useRef<SVGPathElement>(null);
  const rafRef     = useRef<number>(0);
  const astateRef  = useRef<AState>(astate);
  useEffect(() => { astateRef.current = astate; }, [astate]);

  // Clock
  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString("en-US", { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // ESC closes panel
  useEffect(() => {
    const fn = (e: KeyboardEvent) => { if (e.key === "Escape") setPanel(null); };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

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
    return () => { offCompact?.(); offFull?.(); };
  }, []);

  // PRIMARY mode driver — read directly off the real window size.
  // The main process resizes the BrowserWindow (900→90 / 90→900); we mirror
  // that here. This does NOT depend on IPC messages arriving, so the cube
  // reliably replaces the orb even if the enter-compact event is missed.
  useEffect(() => {
    const sync = () => {
      const compact = window.innerWidth < 300 || window.innerHeight < 300;
      setMode(prev => {
        if (compact && prev !== "compact") { setPanel(null); return "compact"; }
        if (!compact && prev !== "full")   { return "full"; }
        return prev;
      });
    };
    sync();
    window.addEventListener("resize", sync);
    return () => window.removeEventListener("resize", sync);
  }, []);

  // Cleanup hover timer
  useEffect(() => () => { if (hoverTimer.current) clearTimeout(hoverTimer.current); }, []);

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
    if (hoverTimer.current) { clearTimeout(hoverTimer.current); hoverTimer.current = null; }
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
            const frac = i / N, a = frac * 2 * Math.PI - Math.PI / 2;
            const noise = Math.sin(frac * Math.PI * 8 + t * 3.2) * amp
              + Math.sin(frac * Math.PI * 15 - t * 2.1) * (amp * .5)
              + Math.sin(frac * Math.PI * 23 + t * 4.5) * (amp * .25);
            const r = CORE + 8 + noise;
            return `${i === 0 ? "M" : "L"}${(CX + r * Math.cos(a)).toFixed(1)} ${(CY + r * Math.sin(a)).toFixed(1)}`;
          });
          waveRef.current.setAttribute("d", pts.join(" ") + "Z");
        }
      }
      if (scanRef.current) {
        const vis = s === "thinking";
        scanRef.current.style.opacity = vis ? "0.65" : "0";
        if (vis) { const a = (t * 95) % 360; scanRef.current.setAttribute("d", sector(CORE - 6, CORE + 20, a - 28, a + 6)); }
      }
      if (speakGRef.current) {
        const vis = s === "speaking";
        speakGRef.current.style.opacity = vis ? "1" : "0";
        if (vis) {
          speakGRef.current.querySelectorAll("line").forEach((line, i) => {
            const b = Math.sin((i / 36) * Math.PI * 7 + t * 9) * 12 + Math.sin((i / 36) * Math.PI * 13 - t * 5) * 6;
            const r1 = CORE + 3, r2 = r1 + Math.max(2, 12 + b);
            const [x1, y1] = polar(r1, i * 10), [x2, y2] = polar(r2, i * 10);
            line.setAttribute("x1", x1.toFixed(1)); line.setAttribute("y1", y1.toFixed(1));
            line.setAttribute("x2", x2.toFixed(1)); line.setAttribute("y2", y2.toFixed(1));
          });
        }
      }
      if (orangeRef.current) {
        if (s === "thinking") { const a = (t * 48) % 360; orangeRef.current.setAttribute("d", sector(122, 136, a - 18, a + 32)); }
        else { const end = s === "idle" ? 5 : s === "listening" ? 52 : 68; orangeRef.current.setAttribute("d", sector(122, 136, -18, end)); }
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const speedMul = astate === "thinking" ? 0.35 : astate === "listening" ? 0.65 : 1;

  const ticks = Array.from({ length: 72 }, (_, i) => {
    const long = i % 3 === 0;
    const [x1, y1] = polar(168, i * 5), [x2, y2] = polar(168 + (long ? 7 : 3), i * 5);
    return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={CYN} strokeWidth={long ? 0.8 : .4} opacity={long ? .45 : .18}/>;
  });

  const panelAngle = MENUS.find(m => m.id === panel)?.angle ?? 0;
  const [clkX, clkY] = polar(235, 315);

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", background: "transparent", overflow: "hidden" }}>
      <style>{`
        @keyframes spin    { to { transform: rotate( 360deg); } }
        @keyframes unspin  { to { transform: rotate(-360deg); } }
        @keyframes breathe { 0%,100%{opacity:.5} 50%{opacity:1} }
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
        onClick={() => { if (mode === "compact") window.electronAPI?.requestRestore(); }}
        style={{
          position: "absolute", inset: 0, zIndex: 200,
          // Clickable ONLY in compact mode — clicking the cube restores the orb.
          pointerEvents: mode === "compact" ? "auto" : "none",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: mode === "compact" ? "pointer" : "default",
        }}
        animate={{
          opacity: mode === "compact" ? 1 : 0,
          scale:   mode === "compact" ? 1 : 0.4,
        }}
        transition={mode === "compact"
          // Cube materialises as the window is still collapsing, so you see a
          // morph rather than a shrink-then-pop.
          ? { opacity: { delay: 0.08, duration: 0.28, ease: "easeOut" },
              scale:   { type: "spring", stiffness: 260, damping: 20, delay: 0.08 } }
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
          position: "absolute", inset: 0,
          pointerEvents: mode === "full" ? "auto" : "none",
        }}
        animate={mode === "full"
          ? { opacity: 1, scale: 1 }
          : { opacity: 0, scale: 0.82 }
        }
        transition={mode === "full"
          // Spring expansion — orb grows in sync with the window expanding
          ? { type: "spring", stiffness: 160, damping: 22, opacity: { duration: 0.3 } }
          : { duration: 0.22, ease: "easeIn" }
        }
      >
        {/* Drag handle — transparent circle over the orb core */}
        <div className="drag-region" style={{
          position: "absolute", left: "50%", top: "50%",
          width: "160px", height: "160px",
          transform: "translate(-50%, -50%)",
          borderRadius: "50%", zIndex: 5, cursor: "move",
        }}/>

        <svg
          viewBox={`0 0 ${VW} ${VH}`}
          width="100%" height="100%"
          className="no-drag"
          style={{ display: "block", position: "absolute", inset: 0 }}
        >
          <defs>
            <radialGradient id="core-g" cx="50%" cy="50%" r="50%">
              <stop offset="0%"   stopColor="#00e5ff" stopOpacity=".22"/>
              <stop offset="45%"  stopColor="#003850" stopOpacity=".3"/>
              <stop offset="100%" stopColor="#000d18" stopOpacity=".98"/>
            </radialGradient>
            <radialGradient id="bg-g" cx="50%" cy="50%" r="50%">
              <stop offset="0%"   stopColor="#00e5ff" stopOpacity=".10"/>
              <stop offset="55%"  stopColor="#00e5ff" stopOpacity=".03"/>
              <stop offset="100%" stopColor="#00e5ff" stopOpacity="0"/>
            </radialGradient>
            <filter id="glow3" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="blur5" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="5"/>
            </filter>
          </defs>

          <ellipse cx={CX} cy={CY} rx={220} ry={210} fill="url(#bg-g)"/>

          {RINGS.map((ring, i) => (
            <circle key={i} cx={CX} cy={CY} r={ring.r} fill="none"
              stroke={i % 2 === 0 ? CYN : CYN2} strokeWidth={ring.w} strokeDasharray={ring.d} opacity={ring.o}
              className="aura-ring"
              style={{ animationName: ring.cw ? "spin" : "unspin", animationDuration: `${ring.s * speedMul}s`, animationTimingFunction: "linear", animationIterationCount: "infinite" }}/>
          ))}
          {ticks}

          <path ref={orangeRef} d={sector(122, 136, -18, 5)} fill={ORG} opacity=".78" filter="url(#glow3)"/>

          {(astate === "listening" || astate === "speaking") && [0, .7, 1.4].map(d => (
            <circle key={d} cx={CX} cy={CY} r={CORE} fill="none" stroke={CYN} strokeWidth="1.5" opacity="0">
              <animate attributeName="r"       from={CORE} to={195} dur="2.2s" begin={`${d}s`} repeatCount="indefinite"/>
              <animate attributeName="opacity" from=".55"  to="0"   dur="2.2s" begin={`${d}s`} repeatCount="indefinite"/>
            </circle>
          ))}

          <circle cx={CX} cy={CY} r={CORE + 18} fill={CYN} opacity=".04" style={{ animation: "breathe 3.5s ease-in-out infinite" }}/>
          <circle cx={CX} cy={CY} r={CORE + 9}  fill={CYN} opacity=".07" style={{ animation: "breathe 3.5s ease-in-out infinite", animationDelay: ".7s" }}/>

          <path ref={waveRef}  fill="none" stroke={CYN} strokeWidth="1.6" style={{ opacity: 0, transition: "opacity .4s", filter: "url(#glow3)" }}/>
          <path ref={scanRef}  fill={CYN} opacity="0" style={{ transition: "opacity .3s", filter: "url(#blur5)" }}/>
          <g    ref={speakGRef} style={{ opacity: 0, transition: "opacity .3s" }}>
            {Array.from({ length: 36 }, (_, i) => <line key={i} stroke={CYN} strokeWidth="1.6" opacity=".6"/>)}
          </g>

          <circle cx={CX} cy={CY} r={CORE}      fill="url(#core-g)" stroke={CYN} strokeWidth="1.4" filter="url(#glow3)" style={{ animation: "breathe 4.2s ease-in-out infinite" }}/>
          <circle cx={CX} cy={CY} r={CORE - 12} fill="none" stroke={CYN} strokeWidth=".4" opacity=".2"/>
          <circle cx={CX} cy={CY} r={CORE - 24} fill="none" stroke={CYN} strokeWidth=".3" opacity=".12"/>
          <line x1={CX - 11} y1={CY}      x2={CX + 11} y2={CY}      stroke={CYN} strokeWidth=".4" opacity=".2"/>
          <line x1={CX}      y1={CY - 11} x2={CX}      y2={CY + 11} stroke={CYN} strokeWidth=".4" opacity=".2"/>
          {[45, 135, 225, 315].map(a => { const [x1,y1]=polar(CORE-3,a),[x2,y2]=polar(CORE-9,a); return <line key={a} x1={x1} y1={y1} x2={x2} y2={y2} stroke={CYN} strokeWidth=".9" opacity=".32"/>; })}

          <text x={CX} y={CY - 12} textAnchor="middle" fill="white" fontSize="17" fontFamily="Orbitron,monospace" fontWeight="700" letterSpacing="6" style={{ animation: "flicker 14s ease-in-out infinite" }}>AURA</text>
          <text x={CX} y={CY + 4}  textAnchor="middle" fill={CYN} fontSize="6.5" fontFamily="'Share Tech Mono',monospace" letterSpacing="2" opacity=".4">v2.6.3</text>
          <text x={CX} y={CY + 20} textAnchor="middle" fill={CYN} fontSize="8.5" fontFamily="'Share Tech Mono',monospace" letterSpacing="2" style={{ animation: "breathe 2s ease-in-out infinite" }}>{STATE_LBL[astate]}</text>

          {/* Clock */}
          <text x={clkX} y={clkY} textAnchor="middle"
            fill={CYN} fontSize="7.5" fontFamily="'Share Tech Mono',monospace" opacity=".38" letterSpacing="1"
          >{clock}</text>


          {/* Radial menu */}
          {MENUS.map(m => {
            const isOpen  = panel   === m.id;
            const isHover = hovering === m.id;
            const R1 = 172, R2 = 208, SPAN = 30;
            const [lx, ly] = polar(220, m.angle);
            return (
              <g key={m.id} className="no-drag" style={{ cursor: "crosshair" }}
                onMouseEnter={() => startHover(m.id)}
                onMouseLeave={() => endHover()}
              >
                <path d={sector(R1, R2, m.angle - SPAN / 2, m.angle + SPAN / 2)} fill="transparent" stroke="none"/>
                <text x={lx} y={ly + 3} textAnchor="middle"
                  fill={isOpen ? CYN : isHover ? "white" : `rgba(0,229,255,0.32)`}
                  fontSize="6.5" fontFamily="'Share Tech Mono',monospace" letterSpacing="1.5"
                  filter={isOpen || isHover ? "url(#glow3)" : undefined}
                  style={{ userSelect: "none", transition: "all .2s ease" }}
                >{m.label}</text>

                {isHover && !isOpen && (
                  <motion.path
                    key={`p-${m.id}`}
                    d={arcOnly(212, m.angle - 15, m.angle + 15)}
                    fill="none" stroke={CYN} strokeWidth="2" strokeLinecap="round"
                    filter="url(#glow3)"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ pathLength: { duration: 0.5, ease: "linear" }, opacity: { duration: 0.08 } }}
                  />
                )}

                {isOpen && (() => {
                  const [dx, dy] = polar(213, m.angle);
                  return <circle cx={dx} cy={dy} r="2" fill={CYN} filter="url(#glow3)" opacity=".9"/>;
                })()}
              </g>
            );
          })}

          {panel && (() => {
            const m = MENUS.find(x => x.id === panel)!;
            return <path d={sector(200, 240, m.angle - 16, m.angle + 16)} fill={CYN} opacity=".07" filter="url(#blur5)"/>;
          })()}

          {/* State controls */}
          <g className="no-drag" transform={`translate(${CX - 148},${VH - 30})`}>
            {(["idle", "listening", "thinking", "speaking"] as AState[]).map((s, i) => {
              const active = astate === s, W = 68, H = 18, G = 8, x = i * (W + G);
              return (
                <g key={s} style={{ cursor: "pointer" }} onClick={() => setAstate(s)}>
                  <rect x={x} y={0} width={W} height={H} rx={3}
                    fill={active ? "rgba(0,229,255,.16)" : "rgba(0,229,255,.03)"}
                    stroke={active ? CYN : "rgba(0,229,255,.14)"} strokeWidth={active ? 1.1 : .4}
                    filter={active ? "url(#glow3)" : undefined}/>
                  <text x={x + W / 2} y={H / 2 + 3} textAnchor="middle"
                    fill={active ? CYN : "rgba(0,229,255,.26)"}
                    fontSize="6" fontFamily="'Share Tech Mono',monospace" letterSpacing="0.5"
                  >{s.toUpperCase()}</text>
                </g>
              );
            })}
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
              style={{ position: "absolute", ...PANEL_POS[panel], width: "260px", maxHeight: "240px", zIndex: 20 }}
            >
              <div className="panel-reveal" style={{
                background: "rgba(0,7,14,0.94)",
                border: "1px solid rgba(0,229,255,0.30)",
                borderRadius: "8px",
                padding: "18px 18px 16px",
                backdropFilter: "blur(18px)",
                WebkitBackdropFilter: "blur(18px)",
                color: CYN,
                fontFamily: "'Share Tech Mono',monospace",
                fontSize: "12px",
                boxShadow: "0 0 40px rgba(0,229,255,.08), inset 0 0 0 1px rgba(0,229,255,.04)",
                overflowY: "auto",
                maxHeight: "240px",
                scrollbarWidth: "none" as const,
              }}>
                <div className="panel-sweep"/>
                <div style={{ position: "absolute", top: 0, left: "10%", right: "10%", height: "1px", background: `linear-gradient(90deg,transparent,${CYN},transparent)`, opacity: .5 }}/>
                {([[0,0,1,1],[1,0,-1,1],[0,1,1,-1],[1,1,-1,-1]] as number[][]).map(([rx,ry,sx,sy], i) => (
                  <div key={i} style={{ position: "absolute", [rx?"right":"left"]: 7, [ry?"bottom":"top"]: 7, width: 12, height: 12,
                    borderTop:    sy === 1  ? "1px solid rgba(0,229,255,.4)" : "none",
                    borderBottom: sy === -1 ? "1px solid rgba(0,229,255,.4)" : "none",
                    borderLeft:   sx === 1  ? "1px solid rgba(0,229,255,.4)" : "none",
                    borderRight:  sx === -1 ? "1px solid rgba(0,229,255,.4)" : "none" }}/>
                ))}
                <PanelContent id={panel}/>
                <button onClick={() => setPanel(null)} style={{
                  position: "absolute", top: 10, right: 10,
                  background: "transparent", border: "1px solid rgba(0,229,255,.28)",
                  color: "rgba(0,229,255,.6)", fontSize: 16, lineHeight: 1,
                  width: 22, height: 22, cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  borderRadius: 3, fontFamily: "monospace",
                }}>×</button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
