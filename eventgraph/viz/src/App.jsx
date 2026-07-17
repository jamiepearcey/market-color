import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const LEGEND = [
  ['company', '#4c9be8'], ['bank', '#56B4E9'], ['central_bank', '#E69F00'],
  ['sovereign', '#22c99a'], ['commodity', '#D55E00'], ['currency', '#CC79A7'],
  ['equity_index', '#F0E442'], ['factor', '#e15759'], ['sector', '#59a14f'],
  ['person', '#9aa0aa'], ['regulator', '#b07aa1'], ['exchange', '#76b7b2'],
  ['rate_or_bond', '#ff9da7'], ['economic_indicator', '#c9a227'], ['other', '#6b7280'],
]
const DIRC = { up: '#22c99a', down: '#f65b5b', widen: '#f65b5b', tighten: '#4c9be8', unchanged: '#8b98a5' }
const eid = e => (typeof e === 'object' ? e.id : e)
const MONTH = 2629800
const ymd = t => new Date(t * 1000).toISOString().slice(0, 7)

function polColor(p) { const h = ((p + 1) / 2) * 140; return `hsl(${h},68%,55%)` }

export default function App() {
  const fg = useRef()
  const [data, setData] = useState({ nodes: [], links: [], sentiments: [], meta: null })
  const [hover, setHover] = useState(null)
  const [sel, setSel] = useState(null)
  const [q, setQ] = useState('')
  const [off, setOff] = useState(() => new Set())
  const [kind, setKind] = useState({ causal: true, sensitivity: true })
  const [colorMode, setColorMode] = useState('type')
  const [range, setRange] = useState(null)   // [startIdx, endIdx] over months
  const [playing, setPlaying] = useState(false)

  useEffect(() => { fetch('/graph.json').then(r => r.json()).then(setData) }, [])

  // month bins for the histogram / slider
  const months = useMemo(() => {
    if (!data.meta) return []
    const out = []; let t = Math.floor(data.meta.tmin / MONTH) * MONTH
    const end = data.meta.tmax
    while (t <= end + MONTH) { out.push({ t0: t, t1: t + MONTH, count: 0 }); t += MONTH }
    data.links.forEach(l => { if (l.ts) { const i = Math.floor((l.ts - out[0].t0) / MONTH); if (out[i]) out[i].count++ } })
    return out
  }, [data])
  useEffect(() => { if (months.length && !range) setRange([0, months.length - 1]) }, [months, range])

  const win = useMemo(() => range && months.length ? [months[range[0]].t0, months[range[1]].t1] : [0, 9e9], [range, months])
  const nodeById = useMemo(() => new Map(data.nodes.map(n => [n.id, n])), [data])

  const linkVisible = useCallback(l => {
    if (!kind[l.kind]) return false
    if (l.ts && (l.ts < win[0] || l.ts > win[1])) return false
    const s = nodeById.get(eid(l.source)), t = nodeById.get(eid(l.target))
    return !!(s && t && !off.has(s.type) && !off.has(t.type))
  }, [kind, win, off, nodeById])

  const visNodes = useMemo(() => {
    const s = new Set(); data.links.forEach(l => { if (linkVisible(l)) { s.add(eid(l.source)); s.add(eid(l.target)) } }); return s
  }, [data, linkVisible])

  const adj = useMemo(() => {
    const m = new Map()
    data.links.forEach(l => { if (!linkVisible(l)) return; const s = eid(l.source), t = eid(l.target); (m.get(s) || m.set(s, new Set()).get(s)).add(t); (m.get(t) || m.set(t, new Set()).get(t)).add(s) })
    return m
  }, [data, linkVisible])

  const focus = hover || sel
  const near = useMemo(() => { if (!focus) return null; const s = new Set([focus.id]); (adj.get(focus.id) || []).forEach(x => s.add(x)); return s }, [focus, adj])

  const sentByNode = useMemo(() => {
    if (colorMode !== 'sentiment') return null
    const acc = new Map()
    data.sentiments.forEach(s => { if (s.p != null && s.ts && s.ts >= win[0] && s.ts <= win[1]) { const a = acc.get(s.id) || [0, 0]; a[0] += s.p; a[1]++; acc.set(s.id, a) } })
    const r = new Map(); acc.forEach(([sum, c], id) => r.set(id, sum / c)); return r
  }, [colorMode, win, data])

  const colorOf = useCallback(n => {
    if (colorMode === 'sentiment') { const p = sentByNode?.get(n.id); return p === undefined ? '#39414d' : polColor(p) }
    return n.color
  }, [colorMode, sentByNode])

  const nodePaint = useCallback((node, ctx, scale) => {
    if (!visNodes.has(node.id)) return
    const dim = near && !near.has(node.id)
    ctx.globalAlpha = dim ? 0.1 : 1
    ctx.beginPath(); ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI); ctx.fillStyle = colorOf(node); ctx.fill()
    if (focus && node.id === focus.id) { ctx.lineWidth = 2 / scale; ctx.strokeStyle = '#fff'; ctx.stroke() }
    if (!dim && (node.degree >= 16 || (near && near.has(node.id) && scale > 1.1))) {
      const fs = Math.min(13, 11 / scale); ctx.font = `600 ${fs}px -apple-system, sans-serif`
      ctx.fillStyle = '#e6edf3'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.globalAlpha = 1
      ctx.fillText(node.name, node.x, node.y - node.val - fs * 0.75)
    }
    ctx.globalAlpha = 1
  }, [visNodes, near, focus, colorOf])

  const linkColor = useCallback(l => {
    const base = l.kind === 'sensitivity' ? '#e15759' : (DIRC[l.direction] || '#8b98a5')
    if (!near) return base + '3a'
    return near.has(eid(l.source)) && near.has(eid(l.target)) ? base + 'dd' : base + '0a'
  }, [near])

  // play: slide the window forward
  useEffect(() => {
    if (!playing || !months.length) return
    const w = range[1] - range[0]
    const id = setInterval(() => setRange(([a, b]) => b >= months.length - 1 ? [0, w] : [a + 1, b + 1]), 550)
    return () => clearInterval(id)
  }, [playing, months, range])

  const search = () => {
    const s = q.trim().toLowerCase(); if (!s) return
    const n = data.nodes.find(n => visNodes.has(n.id) && (n.name.toLowerCase().includes(s) || n.id.toLowerCase().includes(s)))
    if (n && fg.current) { setSel(n); fg.current.centerAt(n.x, n.y, 700); fg.current.zoom(3.2, 700) }
  }

  const selEdges = useMemo(() => {
    if (!sel) return { out: [], inc: [] }
    const out = [], inc = []
    data.links.forEach(l => { if (!linkVisible(l)) return; const s = eid(l.source), t = eid(l.target); if (s === sel.id) out.push({ ...l, other: t }); else if (t === sel.id) inc.push({ ...l, other: s }) })
    return { out, inc }
  }, [sel, data, linkVisible])
  const nameOf = id => nodeById.get(id)?.name || id
  const visCausal = useMemo(() => data.links.filter(l => l.kind === 'causal' && linkVisible(l)).length, [data, linkVisible])
  const visSens = useMemo(() => data.links.filter(l => l.kind === 'sensitivity' && linkVisible(l)).length, [data, linkVisible])
  const maxBar = Math.max(1, ...months.map(m => m.count))

  return (
    <div style={{ position: 'fixed', inset: 0 }}>
      <ForceGraph2D
        ref={fg} graphData={data} backgroundColor="#0d1117"
        nodeRelSize={1} nodeVal="val" nodeCanvasObject={nodePaint}
        nodeVisibility={n => visNodes.has(n.id)}
        nodePointerAreaPaint={(node, color, ctx) => { if (!visNodes.has(node.id)) return; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(node.x, node.y, node.val + 2, 0, 2 * Math.PI); ctx.fill() }}
        linkVisibility={linkVisible}
        linkColor={linkColor}
        linkLineDash={l => (l.kind === 'sensitivity' ? [3, 3] : null)}
        linkWidth={l => {
          const hl = near && near.has(eid(l.source)) && near.has(eid(l.target))
          if (l.kind === 'sensitivity' && l.beta_star != null) return (hl ? 1.6 : 0.7) * (0.5 + Math.min(Math.abs(l.beta_star), 1.5))
          return hl ? 1.3 : 0.4
        }}
        linkDirectionalArrowLength={3} linkDirectionalArrowRelPos={1}
        onNodeHover={setHover} onNodeClick={n => setSel(n)} onBackgroundClick={() => setSel(null)}
        cooldownTicks={140} d3VelocityDecay={0.28} warmupTicks={30}
      />

      <div className="panel left">
        <h1>eventgraph</h1>
        <div className="sub">Bloomberg 2006–2013 · {data.nodes.length.toLocaleString()} entities · causal + sensitivity graph</div>
        <div className="searchrow">
          <input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} placeholder="find an entity (greece, oil, fed…)" />
          <button onClick={search}>↵</button>
        </div>
        <div className="legend">
          {LEGEND.map(([t, c]) => (
            <button key={t} className={off.has(t) ? 'lg off' : 'lg'} onClick={() => setOff(p => { const n = new Set(p); n.has(t) ? n.delete(t) : n.add(t); return n })}>
              <i style={{ background: c }} />{t}
            </button>
          ))}
        </div>
        <div className="hint">scroll = zoom · drag = pan · hover to trace · click a node for its causal + sensitivity edges. Solid = causal (green up / red down), dashed = sensitivity.</div>
      </div>

      {sel && (
        <div className="panel right">
          <button className="x" onClick={() => setSel(null)}>×</button>
          <h2>{sel.name}</h2>
          <div className="tags">
            <span className="tag" style={{ borderColor: sel.color, color: sel.color }}>{sel.type}</span>
            {sel.ticker && <span className="tag tick">{sel.ticker}</span>}
            <span className="tag">{sel.degree} edges</span>
          </div>
          <Section title={`caused / affected →  (${selEdges.out.length})`} items={selEdges.out} nameOf={nameOf} onPick={setSel} nodeById={nodeById} />
          <Section title={`driven by ←  (${selEdges.inc.length})`} items={selEdges.inc} nameOf={nameOf} onPick={setSel} nodeById={nodeById} />
        </div>
      )}

      {months.length > 0 && range && (
        <div className="panel time">
          <div className="timehead">
            <button className="play" onClick={() => setPlaying(p => !p)}>{playing ? '⏸' : '▶'}</button>
            <span className="win">{months[range[0]] && ymd(months[range[0]].t0)} — {months[range[1]] && ymd(months[range[1]].t0)}</span>
            <span className="cnt">{visCausal.toLocaleString()} causal · {visSens.toLocaleString()} sensitivity edges in window</span>
            <span className="ctl">
              <button className={kind.causal ? 'chip' : 'chip off'} onClick={() => setKind(k => ({ ...k, causal: !k.causal }))}><i style={{ background: '#22c99a' }} />causal</button>
              <button className={kind.sensitivity ? 'chip' : 'chip off'} onClick={() => setKind(k => ({ ...k, sensitivity: !k.sensitivity }))}><i className="dash" />sensitivity</button>
              <select className="mode" value={colorMode} onChange={e => setColorMode(e.target.value)}>
                <option value="type">color: type</option>
                <option value="sentiment">color: sentiment</option>
              </select>
            </span>
          </div>
          <div className="hist">
            {months.map((m, i) => (
              <div key={i} className={i >= range[0] && i <= range[1] ? 'bar on' : 'bar'} style={{ height: `${100 * m.count / maxBar}%` }} title={`${ymd(m.t0)}: ${m.count}`} />
            ))}
          </div>
          <div className="sliders">
            <input type="range" min="0" max={months.length - 1} value={range[0]} onChange={e => setRange(([a, b]) => [Math.min(+e.target.value, b), b])} />
            <input type="range" min="0" max={months.length - 1} value={range[1]} onChange={e => setRange(([a, b]) => [a, Math.max(+e.target.value, a)])} />
          </div>
        </div>
      )}
    </div>
  )
}

function Section({ title, items, nameOf, onPick, nodeById }) {
  if (!items.length) return null
  return (
    <div className="section">
      <h3>{title}</h3>
      {items.slice(0, 60).map((l, i) => (
        <div className="edge" key={i} style={{ borderLeftColor: l.kind === 'sensitivity' ? '#e15759' : (DIRC[l.direction] || '#30363d') }}>
          <div className="edgehead">
            <b style={{ cursor: 'pointer' }} onClick={() => onPick(nodeById.get(l.other))}>{nameOf(l.other)}</b>{' '}
            <span className="mech">{l.kind === 'sensitivity'
              ? (l.beta_star != null ? `sensitivity · measured β*=${l.beta_star.toFixed(2)}${l.r2 != null ? ` R²=${l.r2}` : ''} ${l.agrees ? '✓' : '✗ vs narrative'}` : `sensitivity · ${l.direction} · ${l.basis || ''}`)
              : `${l.mechanism} · ${l.direction} · ${l.modality}`}{l.date ? ` · ${l.date}` : ''}</span>
          </div>
          {l.quote && <div className="quote">“{l.quote}”</div>}
        </div>
      ))}
    </div>
  )
}
