# Copied UI Inventory

This inventory keeps the QuantFabric donor app available while the product
shifts toward a notebook-first quant brain. The intent is to preserve useful UX
patterns and park platform-heavy surfaces until they prove necessary.

## Keep

| Area | Files | Why |
| --- | --- | --- |
| Notebook workbench | `ExploreView.tsx`, `ExploreSidebar.tsx`, `NotebookCell.tsx`, `CodeEditor.tsx` | The closest existing shape to the new product: ordered cells, SQL/markdown editing, file-backed notebooks, run output, and promotion paths. |
| Result inspection | `OutputTable.tsx` | DataTable-style result browsing is core to analytical replay. Keep the table UX even if the backing grid dependency changes. |
| Shell primitives | `Header.tsx`, `PrimaryRail.tsx`, `PageHeader`, `Button`, `Badge`, `Tabs`, `ScrollArea` | Useful app chrome and interaction vocabulary. Rename and simplify before deleting. |
| Notifications and confirmation | `Toaster.tsx`, `ConfirmDialog.tsx` | Replay/fork/schedule actions need clear action feedback. |
| Settings and secrets | `SettingsView.tsx` | Secret custody still matters once the brain calls external LLM/tool providers. |

## Reshape

| Area | Files | New Role |
| --- | --- | --- |
| Workspace dashboard | `WorkspaceView.tsx` | Becomes a notebook landing surface or recent task dashboard if useful. The new default is `BrainNotebookView`. |
| Connectors and bindings | `ConnectorsView.tsx`, `BindingsView.tsx`, `ConnectorCombobox.tsx`, `FilePathDropzone.tsx` | Reframe as Data Setup: source -> validation -> snapshot -> notebook input. |
| Jobs and schedules | `JobsView.tsx`, `CronEditor.tsx`, `job-readiness.ts` | Reframe as scheduled notebook/task replays, not a distributed flow editor. |
| Packs/templates | `PacksView.tsx`, `StudioView.tsx`, `TemplateCombobox.tsx`, `TomlView.tsx` | Keep as reusable task blocks and power-user authoring, not the default abstraction. |
| AI panel | `AiPanel.tsx` | Move from sidecar chat to task-cell planning and critique inside the notebook. |

## Park

| Area | Files | Reason |
| --- | --- | --- |
| Clusters | `ClustersView.tsx` | Useful donor UX for future execution targets, but not part of the initial notebook product. |
| Observability | `ObservabilityView.tsx` | Keep the operational patterns; hide from the primary product path until scheduled runs exist. |
| Staged relations | `StagedRelationsView.tsx` | The artifact idea is important. The current control-plane shape is too platform-specific for the first pass. |
| Bootstrap/admin | `BootstrapWizard.tsx` | Platform setup flow, not notebook IP. |

## Delete Later

Nothing should be deleted in the first redesign pass. Removal should happen
only after a replacement path exists or a surface is confirmed to be dead weight.

## First Navigation Shape

- Notebook: brain notebook and donor Explore workbench.
- Data: connectors, bindings, staged artifacts.
- Runs: jobs, packs, templates.
- Parked: workspace, clusters, observability.
- Settings.

## UX IP To Protect

- Prompt, plan, deterministic spec, result, caveat, and replay manifest live in
  the same notebook flow.
- A user can fork from any result cell without losing the original manifest.
- The LLM is visible as an operator that proposes specs and critiques results,
  not as the source of numbers.
- Every displayed number points back to a deterministic replay artifact.
