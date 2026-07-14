import { describe, expect, it } from "vitest";
import { layoutGraph, nodeRadius } from "./graph-layout";
import type { Graph } from "./mc-api";

const GRAPH: Graph = {
  nodes: [
    { id: "oil", count: 10 },
    { id: "hormuz", count: 6 },
    { id: "lng", count: 2 },
    { id: "pakistan", count: 1 },
  ],
  edges: [
    { source: "oil", target: "hormuz", weight: 5, causal: false },
    { source: "lng", target: "pakistan", weight: 1, causal: false },
    { source: "hormuz", target: "oil", weight: 2, causal: true },
  ],
};

describe("layoutGraph", () => {
  it("produces finite, in-bounds positions for every node and resolves edge endpoints", () => {
    const layout = layoutGraph(GRAPH);
    expect(layout.nodes).toHaveLength(4);
    expect(layout.edges).toHaveLength(3);
    for (const n of layout.nodes) {
      expect(Number.isFinite(n.x)).toBe(true);
      expect(Number.isFinite(n.y)).toBe(true);
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.y).toBeGreaterThanOrEqual(0);
      expect(n.x).toBeLessThanOrEqual(layout.width);
      expect(n.y).toBeLessThanOrEqual(layout.height);
    }
    const causal = layout.edges.find((e) => e.causal);
    expect(causal?.source.id).toBe("hormuz");
    expect(causal?.target.id).toBe("oil");
  });

  it("is deterministic for the same graph and sizes nodes by count", () => {
    const a = layoutGraph(GRAPH);
    const b = layoutGraph(GRAPH);
    expect(a.nodes.map((n) => [n.x, n.y])).toEqual(b.nodes.map((n) => [n.x, n.y]));
    expect(nodeRadius(10)).toBeGreaterThan(nodeRadius(1));
    expect(a.nodes[0].r).toBeGreaterThan(a.nodes[3].r);
  });

  it("spreads connected nodes apart instead of collapsing them", () => {
    const layout = layoutGraph(GRAPH);
    const [oil, hormuz] = [layout.nodes[0], layout.nodes[1]];
    const d = Math.hypot(oil.x - hormuz.x, oil.y - hormuz.y);
    expect(d).toBeGreaterThan(oil.r + hormuz.r); // no overlap between the two hubs
  });

  it("handles an empty graph without NaNs", () => {
    const layout = layoutGraph({ nodes: [], edges: [] });
    expect(layout.nodes).toEqual([]);
    expect(Number.isFinite(layout.width)).toBe(true);
    expect(Number.isFinite(layout.height)).toBe(true);
  });
});
