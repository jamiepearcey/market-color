// Explore notebook (ADR-0029) — pure helpers shared by the store and the view:
// id minting, parameter extraction, preview substitution, and the SQL-cell →
// `[template.sql]` TOML resolution that powers "observe resolved TOML / save /
// add to pack". Kept dependency-free and side-effect-free so they're trivially
// testable and reused identically wherever a notebook is rendered or persisted.
import type {
  CellKind,
  InputKind,
  JobStep,
  JobTemplate,
  Notebook,
  NotebookCell,
  NotebookParam,
  Pack,
  ScalarInputKind,
  Template,
} from "./types";

let seq = 0;
/** Monotonic, collision-resistant id for a freshly-created cell/notebook. */
function mintId(prefix: string): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

export function newCell(kind: CellKind, source = ""): NotebookCell {
  return { id: mintId("cell"), kind, source };
}

/** A starter notebook: a short intro, then a runnable SQL cell. */
export function newNotebook(): Notebook {
  return {
    id: mintId("nb"),
    name: "Untitled exploration",
    source: "",
    params: [],
    cells: [
      newCell(
        "markdown",
        "# New exploration\n\nBind a **data source** above, then write DuckDB SQL against the `input` table. Add `{{placeholders}}` to parameterise a query, then resolve it to a reusable template.",
      ),
      newCell("sql", "SELECT *\nFROM input\nLIMIT 20"),
    ],
  };
}

const TOKEN_RE = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g;

/** Distinct `{{name}}` placeholder names in a SQL string, in first-seen order. */
export function extractParams(sql: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  let m: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(sql))) {
    const name = m[1];
    if (!seen.has(name)) {
      seen.add(name);
      out.push(name);
    }
  }
  return out;
}

/** Every distinct placeholder across all SQL cells of a notebook. */
export function notebookParamNames(nb: Notebook): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const cell of nb.cells) {
    if (cell.kind !== "sql") continue;
    for (const name of extractParams(cell.source)) {
      if (!seen.has(name)) {
        seen.add(name);
        out.push(name);
      }
    }
  }
  return out;
}

/** Quote a SQL identifier (double-quote, doubling embedded quotes). */
function quoteIdent(name: string): string {
  return `"${name.replace(/"/g, '""')}"`;
}

/** Quote a SQL string literal (single-quote, doubling embedded quotes). */
function quoteLiteral(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

/** Render a param's test value for a *preview* run, mirroring the engine's
 *  binding semantics: columns become quoted identifiers, numbers are emitted
 *  bare, dates as quoted literals. */
function renderParamValue(param: NotebookParam): string {
  const raw = param.value.trim();
  switch (param.kind) {
    case "Column":
      return quoteIdent(raw);
    case "Columns":
      return raw
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean)
        .map(quoteIdent)
        .join(", ");
    case "Number":
      return raw === "" ? "NULL" : raw;
    case "Date":
      return quoteLiteral(raw);
    default:
      return quoteLiteral(raw);
  }
}

/** Substitute `{{name}}` placeholders with their param test values for a
 *  preview run. Unknown placeholders are left intact so the engine surfaces a
 *  clear error rather than silently dropping them. */
export function substitute(sql: string, params: NotebookParam[]): string {
  const byName = new Map(params.map((p) => [p.name, p]));
  return sql.replace(TOKEN_RE, (whole, name) => {
    const param = byName.get(name);
    return param ? renderParamValue(param) : whole;
  });
}

/** Whether a SQL cell still has unresolved placeholders without a param. */
export function missingParams(sql: string, params: NotebookParam[]): string[] {
  const known = new Set(params.map((p) => p.name));
  return extractParams(sql).filter((n) => !known.has(n));
}

const KIND_TO_TOML: Record<InputKind, string> = {
  Table: "table",
  TypedRelation: "typed_relation",
  Column: "column",
  Columns: "columns",
  Number: "number",
  Date: "date",
};

/** A filesystem/identifier-safe slug for a template/file name. */
export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 64) || "exploration"
  );
}

/** Encode a string as a TOML basic/multiline-basic string. Single-line values
 *  use JSON-escaping (valid TOML); multiline SQL uses a readable `"""…"""`
 *  block with backslashes and quotes escaped so it round-trips. */
function tomlString(value: string): string {
  if (!value.includes("\n")) return JSON.stringify(value);
  const body = value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  return `"""\n${body}\n"""`;
}

export interface ResolveOptions {
  /** Template name (slugified into the `name` field). */
  name: string;
  description: string;
}

/**
 * Resolve a SQL cell to a complete `[[template]]` + `[template.sql]` TOML
 * document: one `table` input (`data` → the `input` relation), one input per
 * placeholder, the cell SQL with `{{placeholders}}` preserved, and a
 * `[template.example]` wired from the source + param test values so it passes
 * the engine's self-test gate. This is the live "observe resolved TOML".
 */
/** A new user pack assembled from a whole notebook (ADR-0030 P5): one
 *  `[template.sql]` per SQL cell, plus a `*.job.toml` that chains those
 *  templates in order over the notebook's source — so "promote notebook → pack"
 *  is one click and the resulting pack is immediately usable to create a job. */
export interface PackBundleDraft {
  id: string;
  name: string;
  description: string;
  files: { path: string; contents: string }[];
}

export interface NotebookJobDraft {
  name: string;
  description: string;
  source: string;
  steps: JobStep[];
  templates: { name: string; source: string }[];
  includedCells: { id: string; index: number; template: string; source: string }[];
  droppedCells: { id: string; index: number; kind: CellKind; reason: string }[];
  sqlCellCount: number;
}

/** Convert notebook SQL cells into generated templates plus a chained job
 *  definition. Used by both "Create job" and "Promote to pack" so the one-off
 *  and reusable paths stay behaviorally aligned. */
export function toNotebookJobDraft(nb: Notebook): NotebookJobDraft {
  const baseSlug = slugify(nb.name);
  const sqlCells = nb.cells.filter((c) => c.kind === "sql" && c.source.trim());
  const byName = new Map(nb.params.map((p) => [p.name, p]));
  const templates: NotebookJobDraft["templates"] = [];
  const steps: JobStep[] = [];
  const includedCells: NotebookJobDraft["includedCells"] = [];
  const droppedCells: NotebookJobDraft["droppedCells"] = [];

  nb.cells.forEach((cell, index) => {
    if (cell.kind !== "sql") {
      droppedCells.push({
        id: cell.id,
        index,
        kind: cell.kind,
        reason: "Markdown cells document the flow but are not runnable job steps.",
      });
      return;
    }
    if (!cell.source.trim()) {
      droppedCells.push({
        id: cell.id,
        index,
        kind: cell.kind,
        reason: "Empty SQL cells are not runnable job steps.",
      });
      return;
    }
    const stepNumber = includedCells.length + 1;
    const name = `${baseSlug}_s${stepNumber}`;
    templates.push({
      name,
      source: toTemplateToml(cell, nb, {
        name,
        description: `${nb.name} - step ${stepNumber}`,
      }),
    });
    steps.push({
      template: name,
      args: Object.fromEntries(
        extractParams(cell.source).map((ph) => [ph, byName.get(ph)?.value ?? ""]),
      ),
    });
    includedCells.push({ id: cell.id, index, template: name, source: cell.source });
  });

  return {
    name: baseSlug,
    description: `Job derived from notebook "${nb.name}".`,
    source: nb.source,
    steps,
    templates,
    includedCells,
    droppedCells,
    sqlCellCount: sqlCells.length,
  };
}

/** Build a pack bundle from a notebook: SQL cells → templates, chained into a
 *  job template; markdown cells are dropped (they document, they don't run). */
export function toPackBundle(nb: Notebook): PackBundleDraft {
  const baseSlug = slugify(nb.name);
  const id = baseSlug.replace(/_/g, "-");
  const files: { path: string; contents: string }[] = [];
  const draft = toNotebookJobDraft(nb);

  const steps = draft.steps.map((step) => {
    const argPairs = Object.entries(step.args).map(
      ([name, value]) => `${name} = ${JSON.stringify(String(value))}`,
    );
    return `[[job_template.step]]\ntemplate = ${JSON.stringify(step.template)}\nargs = ${
      argPairs.length ? `{ ${argPairs.join(", ")} }` : "{}"
    }`;
  });

  draft.templates.forEach((template) => {
    files.push({
      path: `templates/${template.name}.toml`,
      contents: template.source,
    });
  });

  const job: string[] = [
    "[job_template]",
    `name = ${JSON.stringify(baseSlug)}`,
    `description = ${JSON.stringify(draft.description)}`,
    "",
    "[job_template.source]",
    `file = ${JSON.stringify(draft.source)}`,
  ];
  if (steps.length) {
    job.push("", steps.join("\n\n"));
  }
  files.push({ path: `jobs/${baseSlug}.job.toml`, contents: job.join("\n") + "\n" });
  files.push({
    path: "README.md",
    contents: `# ${nb.name}\n\nPromoted from an Explore notebook - ${draft.sqlCellCount} SQL step${
      draft.sqlCellCount === 1 ? "" : "s"
    } chained into the \`${baseSlug}\` job template.\n`,
  });

  return {
    id,
    name: nb.name,
    description: `Promoted from notebook “${nb.name}”.`,
    files,
  };
}

export function toTemplateToml(cell: NotebookCell, nb: Notebook, opts: ResolveOptions): string {
  const name = slugify(opts.name);
  const placeholders = extractParams(cell.source);
  const byName = new Map(nb.params.map((p) => [p.name, p]));
  const lines: string[] = [];
  lines.push("[[template]]");
  lines.push(`name = ${JSON.stringify(name)}`);
  lines.push(`description = ${JSON.stringify(opts.description || `Exploration: ${name}`)}`);
  lines.push("");
  lines.push("[[template.inputs]]");
  lines.push('name = "data"');
  lines.push('kind = "table"');
  for (const ph of placeholders) {
    const kind = byName.get(ph)?.kind ?? "Column";
    lines.push("[[template.inputs]]");
    lines.push(`name = ${JSON.stringify(ph)}`);
    lines.push(`kind = ${JSON.stringify(KIND_TO_TOML[kind])}`);
  }
  lines.push("");
  lines.push("[template.sql]");
  lines.push(`template = ${tomlString(cell.source.trim())}`);
  lines.push("");
  lines.push("[template.example]");
  const exampleArgs: string[] = [`data = ${JSON.stringify(nb.source)}`];
  for (const ph of placeholders) {
    const val = byName.get(ph)?.value ?? "";
    exampleArgs.push(`${ph} = ${JSON.stringify(val)}`);
  }
  lines.push(`args = { ${exampleArgs.join(", ")} }`);
  return lines.join("\n") + "\n";
}

export function notebookFromPackJobTemplate(
  pack: Pack,
  jobTemplate: JobTemplate,
  templates: readonly Template[],
): Notebook {
  const paramsByName = new Map<string, NotebookParam>();
  for (const input of jobTemplate.inputs) {
    paramsByName.set(input.name, {
      name: input.name,
      kind: normalizeNotebookParamKind(input.kind),
      value: "",
    });
  }

  const cells: NotebookCell[] = [
    newCell(
      "markdown",
      `# ${jobTemplate.name}\n\nPackaged flow from ${pack.name}. Edit SQL cells here, then promote to a new pack or create a job draft.`,
    ),
  ];

  for (const [index, step] of jobTemplate.steps.entries()) {
    const template = templates.find((item) => item.name === step.template);
    for (const [name, value] of Object.entries(step.args ?? {})) {
      if (!paramsByName.has(name)) {
        paramsByName.set(name, {
          name,
          kind: "Column",
          value: String(value),
        });
      } else {
        paramsByName.get(name)!.value = String(value);
      }
    }
    const sql = template?.source ? extractTemplateSql(template.source) : null;
    if (sql) {
      cells.push(newCell("sql", sql));
    } else {
      cells.push(
        newCell(
          "markdown",
          `## Step ${index + 1}: ${step.template}\n\nThis unit template could not be expanded into inline SQL in Explore. Its current args are:\n\n\`\`\`json\n${JSON.stringify(step.args ?? {}, null, 2)}\n\`\`\``,
        ),
      );
    }
  }

  return {
    id: mintId("nb"),
    name: `${pack.name} / ${jobTemplate.name}`,
    source: jobTemplate.source.file || "",
    params: [...paramsByName.values()],
    cells,
  };
}

function normalizeNotebookParamKind(kind: string): ScalarInputKind {
  const lower = kind.toLowerCase();
  if (lower === "columns") return "Columns";
  if (lower === "number") return "Number";
  if (lower === "date") return "Date";
  return "Column";
}

function extractTemplateSql(source: string): string | null {
  const marker = source.search(/^\s*\[template\.sql\]\s*$/m);
  if (marker < 0) return null;
  const rest = source.slice(marker);
  const match = rest.match(/^\s*template\s*=\s*("""[\s\S]*?"""|"(?:\\.|[^"\\])*")/m);
  if (!match) return null;
  const raw = match[1];
  if (raw.startsWith('"""')) {
    return raw.slice(3, -3).replace(/\\"/g, '"').replace(/\\\\/g, "\\").trim();
  }
  try {
    return JSON.parse(raw) as string;
  } catch {
    return null;
  }
}
