// Hand-rolled force-directed layout for the transmission map — no npm deps.
//
// Fruchterman–Reingold-style: pairwise repulsion + spring attraction along
// edges + weak centering, integrated with velocity damping and a cooling
// alpha. Node positions are seeded deterministically from the entity name so
// the same query always settles into the same shape. Sized for the API's
// max_nodes=150 default (O(n²) repulsion is ~22k pair ops per tick).

import type { Graph } from "@/lib/mc-api";

export interface LaidOutNode {
  id: string;
  count: number;
  x: number;
  y: number;
  r: number;
}

export interface LaidOutEdge {
  source: LaidOutNode;
  target: LaidOutNode;
  weight: number;
  causal: boolean;
}

export interface Layout {
  nodes: LaidOutNode[];
  edges: LaidOutEdge[];
  width: number;
  height: number;
}

/** Deterministic hash → [0, 1) for stable initial placement. */
function hash01(s: string, salt: number): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

export function nodeRadius(count: number): number {
  return 4 + Math.sqrt(Math.max(count, 1)) * 2.2;
}

export function layoutGraph(graph: Graph, ticks = 300): Layout {
  const n = graph.nodes.length;
  const spread = Math.max(240, Math.sqrt(n) * 90);

  const nodes: (LaidOutNode & { vx: number; vy: number })[] = graph.nodes.map((node) => {
    const angle = hash01(node.id, 1) * Math.PI * 2;
    const rad = (0.25 + 0.75 * hash01(node.id, 2)) * spread * 0.5;
    return {
      id: node.id,
      count: node.count,
      x: Math.cos(angle) * rad,
      y: Math.sin(angle) * rad,
      r: nodeRadius(node.count),
      vx: 0,
      vy: 0,
    };
  });
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges = graph.edges.flatMap((e) => {
    const source = byId.get(e.source);
    const target = byId.get(e.target);
    return source && target ? [{ source, target, weight: e.weight, causal: e.causal }] : [];
  });

  const k = n > 1 ? spread / Math.sqrt(n) : spread; // ideal pair distance
  let alpha = 1;
  const alphaDecay = 1 - Math.pow(0.005, 1 / Math.max(ticks, 1));

  for (let t = 0; t < ticks && alpha > 0.005; t++) {
    // repulsion
    for (let i = 0; i < n; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < n; j++) {
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1e-4) {
          dx = (hash01(a.id, t) - 0.5) * 0.1;
          dy = (hash01(b.id, t) - 0.5) * 0.1;
          d2 = dx * dx + dy * dy + 1e-4;
        }
        const d = Math.sqrt(d2);
        const f = ((k * k) / d2) * alpha;
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
    }
    // spring attraction along edges (heavier edges pull a little harder)
    for (const e of edges) {
      const dx = e.target.x - e.source.x;
      const dy = e.target.y - e.source.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1e-3;
      const strength = Math.min(1, 0.3 + Math.log2(e.weight + 1) * 0.15);
      const f = ((d - k) / d) * 0.08 * strength * alpha;
      const fx = dx * f;
      const fy = dy * f;
      e.source.vx += fx;
      e.source.vy += fy;
      e.target.vx -= fx;
      e.target.vy -= fy;
    }
    // weak centering + integrate
    for (const node of nodes) {
      node.vx -= node.x * 0.02 * alpha;
      node.vy -= node.y * 0.02 * alpha;
      node.vx *= 0.6;
      node.vy *= 0.6;
      node.x += node.vx;
      node.y += node.vy;
    }
    alpha -= alpha * alphaDecay;
  }

  // normalize into a padded box
  const pad = 40;
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x - node.r);
    minY = Math.min(minY, node.y - node.r);
    maxX = Math.max(maxX, node.x + node.r);
    maxY = Math.max(maxY, node.y + node.r);
  }
  if (!Number.isFinite(minX)) {
    minX = minY = -1;
    maxX = maxY = 1;
  }
  for (const node of nodes) {
    node.x = node.x - minX + pad;
    node.y = node.y - minY + pad;
  }

  return {
    nodes,
    edges,
    width: maxX - minX + pad * 2,
    height: maxY - minY + pad * 2,
  };
}
