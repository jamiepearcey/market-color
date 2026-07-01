import CodeMirror from "@uiw/react-codemirror";
import {
  EditorView,
  keymap,
  Decoration,
  ViewPlugin,
  tooltips,
  type DecorationSet,
  type ViewUpdate,
} from "@codemirror/view";
import { StreamLanguage } from "@codemirror/language";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { sql, SQLite } from "@codemirror/lang-sql";
import { toml } from "@codemirror/legacy-modes/mode/toml";
import { oneDark, oneDarkHighlightStyle } from "@codemirror/theme-one-dark";
import { Prec, RangeSetBuilder, type Extension } from "@codemirror/state";
import {
  autocompletion,
  type Completion,
  type CompletionContext,
} from "@codemirror/autocomplete";
import { highlightTree } from "@lezer/highlight";
import { listQuantFunctions, type QuantFunctionInfo } from "@/lib/api";

/** Language extension chosen by file extension. */
function languageFor(path: string): Extension[] {
  const ext = path.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "toml":
      return [StreamLanguage.define(toml)];
    case "json":
      return [json()];
    case "md":
    case "markdown":
      return [markdown()];
    case "sql":
      return [sql({ dialect: SQLite, upperCaseKeywords: true })];
    default:
      return [];
  }
}

type SourceSqlRange = {
  valueFrom: number;
  valueTo: number;
  stringFrom: number;
  stringTo: number;
  rawString: string;
};

const DUCKDB_SQL_COMPLETIONS: Completion[] = [
  ...[
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "INNER JOIN",
    "CROSS JOIN",
    "ON",
    "AS",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "CAST",
    "FILTER",
    "OVER",
    "PARTITION BY",
    "ROWS BETWEEN",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "IS NULL",
    "IS NOT NULL",
  ].map((label) => ({ label, type: "keyword", section: "DuckDB SQL" })),
  ...[
    "input",
    "list_value",
    "list_count",
    "isfinite",
    "avg",
    "sum",
    "count",
    "min",
    "max",
    "stddev_pop",
    "var_pop",
    "regr_count",
    "regr_avgx",
    "regr_avgy",
    "regr_sxx",
    "regr_syy",
    "regr_sxy",
    "tdigest_from_values",
    "tdigest_merge_states",
    "var_from_tdigest",
    "es_from_tdigest",
    "quantile_from_tdigest",
  ].map((label) => ({
    label,
    type: label === "input" ? "variable" : "function",
    section: "DuckDB built-ins",
  })),
];

function findSourceSqlRanges(source: string): SourceSqlRange[] {
  const ranges: SourceSqlRange[] = [];
  const re = /(^[ \t]*source_sql[ \t]*=[ \t]*)("(?:(?:\\.)|[^"\\])*")/gm;
  let match: RegExpExecArray | null;
  while ((match = re.exec(source))) {
    const prefix = match[1] ?? "";
    const rawString = match[2] ?? "";
    const stringFrom = match.index + prefix.length;
    const stringTo = stringFrom + rawString.length;
    ranges.push({
      valueFrom: stringFrom + 1,
      valueTo: stringTo - 1,
      stringFrom,
      stringTo,
      rawString,
    });
  }
  return ranges;
}

function sourceSqlRangeAt(source: string, pos: number): SourceSqlRange | null {
  return (
    findSourceSqlRanges(source).find(
      (range) => range.valueFrom <= pos && pos <= range.valueTo,
    ) ?? null
  );
}

// --- Embedded SQL syntax highlighting inside TOML `source_sql = "..."` ---
// The TOML stream mode paints the whole value as a single string literal. We
// overlay real DuckDB SQL colours by parsing just the quoted body with the SQL
// grammar and emitting mark decorations through the same oneDark highlight style
// the rest of the editor uses. The raw substring between the quotes is parsed
// directly (source_sql values are single-line TOML basic strings), so highlight
// offsets map 1:1 onto document positions without unescaping. These marks are
// sub-ranges of the TOML string token, so they nest inside it and their colours
// win; punctuation/whitespace the SQL grammar leaves untagged stays string-toned.
const sqlParser = sql({ dialect: SQLite }).language.parser;

function buildSourceSqlDecorations(view: EditorView): DecorationSet {
  const builder = new RangeSetBuilder<Decoration>();
  const source = view.state.doc.toString();
  for (const range of findSourceSqlRanges(source)) {
    if (range.valueTo <= range.valueFrom) continue;
    const sqlText = source.slice(range.valueFrom, range.valueTo);
    const tree = sqlParser.parse(sqlText);
    highlightTree(tree, oneDarkHighlightStyle, (from, to, classes) => {
      if (classes && to > from) {
        builder.add(
          range.valueFrom + from,
          range.valueFrom + to,
          Decoration.mark({ class: classes }),
        );
      }
    });
  }
  return builder.finish();
}

const sourceSqlHighlighter = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) {
      this.decorations = buildSourceSqlDecorations(view);
    }
    update(update: ViewUpdate) {
      if (update.docChanged)
        this.decorations = buildSourceSqlDecorations(update.view);
    }
  },
  { decorations: (v) => v.decorations },
);

function decodeTomlBasicString(rawString: string): string | null {
  try {
    return JSON.parse(rawString) as string;
  } catch {
    return null;
  }
}

function encodeTomlBasicString(value: string): string {
  return JSON.stringify(value);
}

function formatDuckDbSql(value: string): string {
  const keywords = [
    "select",
    "from",
    "where",
    "group by",
    "having",
    "order by",
    "limit",
    "offset",
    "union all",
    "union",
    "left join",
    "right join",
    "full join",
    "inner join",
    "cross join",
    "join",
    "on",
    "and",
    "or",
    "as",
    "case",
    "when",
    "then",
    "else",
    "end",
  ];
  let out = value.trim().replace(/\s+/g, " ");
  for (const keyword of keywords.sort((a, b) => b.length - a.length)) {
    out = out.replace(
      new RegExp(`\\b${keyword.replace(/\s+/g, "\\s+")}\\b`, "gi"),
      keyword.toUpperCase(),
    );
  }
  for (const keyword of [
    "FROM",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "LIMIT",
    "OFFSET",
    "UNION ALL",
    "UNION",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "INNER JOIN",
    "CROSS JOIN",
    "JOIN",
  ]) {
    out = out.replace(
      new RegExp(`\\s+${keyword.replace(/\s+/g, "\\s+")}\\b`, "g"),
      `\n${keyword}`,
    );
  }
  out = out.replace(/^SELECT\s+/i, "SELECT ");
  return out;
}

/** A registered quant function as an autocomplete entry: shows its signature,
 *  groups by model/category, and inserts `name()` with the cursor between the
 *  parentheses ready for arguments. */
function quantFunctionCompletion(fn: QuantFunctionInfo): Completion {
  const signature = `(${fn.args.join(", ")})`;
  return {
    label: fn.name,
    type: "function",
    section: fn.category ? `Celeritas · ${fn.category}` : "Celeritas functions",
    detail: signature,
    ...(fn.returns ? { info: `${fn.name}${signature} → ${fn.returns}` } : {}),
    apply: (
      view: EditorView,
      _completion: Completion,
      from: number,
      to: number,
    ) => {
      view.dispatch({
        changes: { from, to, insert: `${fn.name}()` },
        selection: { anchor: from + fn.name.length + 1 },
      });
    },
  };
}

// The catalog is fetched from the backend once per session and reused across
// every completion request (it does not change while the app is running).
let quantCompletionsPromise: Promise<Completion[]> | null = null;
function quantCompletions(): Promise<Completion[]> {
  if (!quantCompletionsPromise) {
    quantCompletionsPromise = listQuantFunctions()
      .then((fns) => fns.map(quantFunctionCompletion))
      .catch(() => []);
  }
  return quantCompletionsPromise;
}

async function completeSourceSql(context: CompletionContext) {
  const source = context.state.doc.toString();
  if (!sourceSqlRangeAt(source, context.pos)) return null;
  const word = context.matchBefore(/[A-Za-z_][\w.]*/);
  if (!word && !context.explicit) return null;
  const quant = await quantCompletions();
  return {
    from: word?.from ?? context.pos,
    options: [...DUCKDB_SQL_COMPLETIONS, ...quant],
    validFor: /^[A-Za-z_][\w.]*$/,
  };
}

function formatSourceSqlAtCursor(view: EditorView): boolean {
  const source = view.state.doc.toString();
  const range = sourceSqlRangeAt(source, view.state.selection.main.head);
  if (!range) return false;
  const sqlValue = decodeTomlBasicString(range.rawString);
  if (sqlValue == null) return false;
  view.dispatch({
    changes: {
      from: range.stringFrom,
      to: range.stringTo,
      insert: encodeTomlBasicString(formatDuckDbSql(sqlValue)),
    },
  });
  return true;
}

/** Whole-document DuckDB completions for a standalone `.sql` file (the Explore
 *  notebook's SQL cells): keywords, built-ins, and every registered quant UDF —
 *  the same catalog the TOML `source_sql` editor offers, but active everywhere
 *  rather than only inside a quoted `source_sql` value. */
async function completeDuckSql(context: CompletionContext) {
  const word = context.matchBefore(/[A-Za-z_][\w.]*/);
  if (!word && !context.explicit) return null;
  const quant = await quantCompletions();
  return {
    from: word?.from ?? context.pos,
    options: [...DUCKDB_SQL_COMPLETIONS, ...quant],
    validFor: /^[A-Za-z_][\w.]*$/,
  };
}

function sqlNotebookExtensions(path: string): Extension[] {
  if (path.split(".").pop()?.toLowerCase() !== "sql") return [];
  return [autocompletion({ override: [completeDuckSql] })];
}

function functionNameAtCursor(source: string, pos: number): string | null {
  let depth = 0;
  for (let i = Math.min(pos - 1, source.length - 1); i >= 0; i--) {
    const ch = source[i];
    if (ch === ")") {
      depth++;
      continue;
    }
    if (ch !== "(") continue;
    if (depth > 0) {
      depth--;
      continue;
    }
    const end = i;
    let start = end;
    while (start > 0 && /[\w.]/.test(source[start - 1])) start--;
    const name = source.slice(start, end).trim();
    return /^[A-Za-z_][\w.]*$/.test(name) ? name : null;
  }
  const before = source.slice(0, pos).match(/([A-Za-z_][\w.]*)\s*$/);
  return before?.[1] ?? null;
}

function cursorFunctionExtension(
  path: string,
  onFunctionContextChange?: (name: string | null) => void,
): Extension[] {
  if (
    !onFunctionContextChange ||
    path.split(".").pop()?.toLowerCase() !== "sql"
  )
    return [];
  return [
    EditorView.updateListener.of((update) => {
      if (!update.docChanged && !update.selectionSet && !update.focusChanged)
        return;
      const source = update.state.doc.toString();
      onFunctionContextChange(
        functionNameAtCursor(source, update.state.selection.main.head),
      );
    }),
  ];
}

function sourceSqlExtensions(path: string): Extension[] {
  if (path.split(".").pop()?.toLowerCase() !== "toml") return [];
  return [
    sourceSqlHighlighter,
    autocompletion({ override: [completeSourceSql] }),
    Prec.highest(
      keymap.of([
        { key: "Mod-Shift-f", run: formatSourceSqlAtCursor },
        { key: "Alt-Shift-f", run: formatSourceSqlAtCursor },
      ]),
    ),
  ];
}

/** Blend CodeMirror into the app's panel: transparent surface, app type sizing,
 *  no focus ring (the panel already frames it). oneDark supplies the syntax
 *  colours; we only override chrome. */
/** Render tooltips (the autocomplete popover, signature hints) in `document.body`
 *  with fixed positioning so they escape a cell's `overflow` clipping instead of
 *  being cut off inside it, and stay anchored when the notebook cell list
 *  scrolls. */
const escapingTooltips = tooltips({
  ...(typeof document !== "undefined" ? { parent: document.body } : {}),
  position: "fixed",
});

const appTheme = EditorView.theme({
  "&": { backgroundColor: "transparent", height: "100%", fontSize: "12px" },
  ".cm-scroller": {
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace",
    lineHeight: "1.6",
  },
  ".cm-gutters": {
    backgroundColor: "transparent",
    border: "none",
    color: "hsl(var(--muted-foreground))",
  },
  ".cm-content": { padding: "12px 0" },
  "&.cm-focused": { outline: "none" },
  ".cm-activeLine, .cm-activeLineGutter": {
    backgroundColor: "hsl(var(--surface-hover) / 0.4)",
  },
});

/**
 * A CodeMirror-backed code editor for the pack IDE: syntax highlighting + line
 * numbers, read-only support, and ⌘/Ctrl-S to save.
 */
export function CodeEditor({
  path,
  value,
  readOnly,
  onChange,
  onSave,
  onFunctionContextChange,
}: {
  path: string;
  value: string;
  readOnly?: boolean;
  onChange: (value: string) => void;
  onSave?: () => void;
  onFunctionContextChange?: (name: string | null) => void;
}) {
  const saveKeymap = Prec.highest(
    keymap.of([
      {
        key: "Mod-s",
        run: () => {
          onSave?.();
          return true;
        },
      },
    ]),
  );
  return (
    <CodeMirror
      value={value}
      height="100%"
      theme={oneDark}
      {...(readOnly !== undefined ? { readOnly } : {})}
      basicSetup={{
        lineNumbers: true,
        foldGutter: false,
        highlightActiveLine: !readOnly,
      }}
      extensions={[
        saveKeymap,
        escapingTooltips,
        ...languageFor(path),
        ...sourceSqlExtensions(path),
        ...sqlNotebookExtensions(path),
        ...cursorFunctionExtension(path, onFunctionContextChange),
        appTheme,
      ]}
      onChange={onChange}
      className="h-full"
    />
  );
}
