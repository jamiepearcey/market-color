import { Component, useMemo, type ReactNode } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "∅";
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(6).replace(/\.?0+$/, "");
  return String(v);
}

export function OutputTable({
  rows,
}: {
  rows: readonly Record<string, unknown>[];
}) {
  if (!rows.length)
    return (
      <div className="p-6 text-sm text-muted-foreground">No rows returned.</div>
    );
  return (
    <GridResultBoundary rows={rows}>
      <FallbackOutputTable rows={rows} />
    </GridResultBoundary>
  );
}

class GridResultBoundary extends Component<
  { rows: readonly Record<string, unknown>[]; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidUpdate(previousProps: {
    rows: readonly Record<string, unknown>[];
  }) {
    if (previousProps.rows !== this.props.rows && this.state.failed) {
      this.setState({ failed: false });
    }
  }

  render() {
    if (this.state.failed)
      return <FallbackOutputTable rows={this.props.rows} />;
    return this.props.children;
  }
}

function FallbackOutputTable({
  rows,
}: {
  rows: readonly Record<string, unknown>[];
}) {
  const cols = useMemo(() => collectColumns(rows), [rows]);
  return (
    <ScrollArea className="max-h-[340px]">
      <table className="w-full border-collapse text-[12px]">
        <thead className="sticky top-0 bg-surface-header">
          <tr className="border-b border-outline-subtle">
            {cols.map((c) => (
              <th
                key={c}
                className="px-3 py-2 text-left font-mono font-medium text-muted-foreground"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className="border-b border-outline-subtle/70 hover:bg-surface-hover"
            >
              {cols.map((c) => (
                <td
                  key={c}
                  className="px-3 py-1.5 font-mono tabular-nums text-foreground/90"
                >
                  {fmt(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollArea>
  );
}

function collectColumns(rows: readonly Record<string, unknown>[]) {
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) seen.add(key);
  }
  return [...seen];
}
