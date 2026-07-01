import { useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import {
  Search,
  CircleX,
  Loader2,
  Store,
  Download,
  CircleCheck,
  ArrowDownToLine,
  ArrowUpFromLine,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PageHeader } from "@/components/ui/page-header";
import { actions, store, type RegistryTypeFilter } from "@/lib/store";
import { cn } from "@/lib/utils";
import { iconForCategory } from "@/lib/connector-icons";
import type { RegistryPackage } from "@/lib/types";
import { PackageDetailView } from "@/components/PackageDetailView";
import { EmptyPanel, ErrorPanel, LoadingPanel, SkeletonRows } from "@/components/AsyncStates";
import { SimulatedNotice } from "@/components/ui/simulated";

export function StoreView() {
  const snap = useSnapshot(store);
  const [searchDraft, setSearchDraft] = useState(snap.registrySearch);

  useEffect(() => setSearchDraft(snap.registrySearch), [snap.registrySearch]);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (searchDraft !== snap.registrySearch) actions.setRegistrySearch(searchDraft);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchDraft, snap.registrySearch]);

  // Full-details page when a package is selected.
  if (snap.selectedPackage) {
    return <PackageDetailView pkg={snap.selectedPackage as RegistryPackage} />;
  }

  const packages = snap.registryPackages as RegistryPackage[];
  const categories = Array.from(new Set(packages.map((p) => p.category))).sort();

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <PageHeader
        icon={<Store className="size-4 text-primary" />}
        title="Connector Store"
        meta={
          <span className="truncate text-[11px] text-muted-foreground">
            Discover and install additional sources &amp; targets from the online registry.
          </span>
        }
        actions={
          <Button
            size="sm"
            variant="outline"
            onClick={() => actions.loadRegistry()}
            disabled={snap.registryLoading}
          >
            {snap.registryLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            Refresh
          </Button>
        }
      />

      {/* Search + filter toolbar */}
      <div className="flex flex-col gap-2 border-b border-outline-subtle bg-surface-header px-3 py-2">
        <div className="flex items-center gap-2">
          <div className="flex min-w-0 flex-1 items-center gap-1.5 rounded-md border border-outline-subtle bg-surface-input px-2">
            <Search className="size-3.5 shrink-0 text-muted-foreground" />
            <input
              value={searchDraft}
              onChange={(e) => setSearchDraft(e.target.value)}
              placeholder="Search the store by name, tag, or description"
              aria-label="Search the store"
              className="h-7 min-w-0 flex-1 bg-transparent text-[12px] outline-none placeholder:text-muted-foreground"
            />
            {searchDraft && (
              <button
                onClick={() => setSearchDraft("")}
                className="text-muted-foreground hover:text-foreground"
                title="Clear search"
                aria-label="Clear search"
              >
                <CircleX className="size-3.5" />
              </button>
            )}
          </div>
          <TypeChip value="all" label="All" />
          <TypeChip value="source" label="Sources" icon={ArrowDownToLine} />
          <TypeChip value="target" label="Targets" icon={ArrowUpFromLine} />
        </div>

        {categories.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <CategoryChip value="all" label="All categories" />
            {categories.map((c) => (
              <CategoryChip key={c} value={c} label={c} />
            ))}
          </div>
        )}
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="p-6">
          <SimulatedNotice className="mb-4">
            Preview mode: store results come from the bundled offline registry seed, not a live sidecar registry fetch.
          </SimulatedNotice>
          {snap.registryLoading && packages.length === 0 ? (
            <SkeletonRows count={6} />
          ) : snap.registryError ? (
            <ErrorPanel
              title="Could not load the connector store"
              message={snap.registryError}
              onRetry={() => actions.loadRegistry()}
            />
          ) : packages.length === 0 ? (
            <EmptyPanel
              title="No packages found"
              message="Try a different search term or broaden the current filter."
              action={
                <Button size="sm" variant="outline" onClick={() => actions.loadRegistry()}>
                  <RefreshCw className="size-3.5" /> Refresh
                </Button>
              }
            />
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {packages.map((p) => (
                <PackageCard key={p.name} pkg={p} installing={Boolean(snap.installing[p.name])} />
              ))}
            </div>
          )}
          {snap.registryLoading && packages.length > 0 ? (
            <LoadingPanel label="Refreshing the registry…" className="mt-4 py-6" />
          ) : null}
        </div>
      </ScrollArea>
    </div>
  );
}

function TypeChip({
  value,
  label,
  icon: Icon,
}: {
  value: RegistryTypeFilter;
  label: string;
  icon?: typeof ArrowDownToLine;
}) {
  const snap = useSnapshot(store);
  const active = snap.registryTypeFilter === value;
  return (
    <button
      onClick={() => actions.setRegistryTypeFilter(value)}
      aria-pressed={active}
      className={cn(
        "flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors",
        active
          ? "border-primary/40 bg-primary/[0.12] text-foreground"
          : "border-outline-subtle text-muted-foreground hover:bg-surface-hover hover:text-foreground",
      )}
    >
      {Icon && <Icon className="size-3.5" />} {label}
    </button>
  );
}

function CategoryChip({ value, label }: { value: string; label: string }) {
  const snap = useSnapshot(store);
  const active = snap.registryCategoryFilter === value;
  return (
    <button
      onClick={() => actions.setRegistryCategoryFilter(value)}
      aria-pressed={active}
      className={cn(
        "rounded-md border px-2 py-0.5 text-[10.5px] transition-colors",
        active
          ? "border-primary/40 bg-primary/[0.12] text-foreground"
          : "border-outline-subtle text-muted-foreground hover:bg-surface-hover hover:text-foreground",
      )}
    >
      {label}
    </button>
  );
}

function PackageCard({ pkg, installing }: { pkg: RegistryPackage; installing: boolean }) {
  const Icon = iconForCategory(pkg.category);
  const isSource = pkg.type === "source";
  return (
    <div className="group flex flex-col rounded-md border border-outline-subtle bg-surface-panel p-3 transition-colors hover:border-primary/[0.25]">
      <button
        onClick={() => actions.selectPackage(pkg.name)}
        className="text-left"
        title="View full details"
      >
        <div className="flex items-center gap-2">
          <span className="grid size-7 shrink-0 place-items-center rounded border border-transparent bg-primary/[0.14]">
            <Icon className="size-3.5 text-icon-tile-foreground" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] font-semibold tracking-[-0.015em]">
              {pkg.title}
            </span>
            <span className="mt-0.5 flex items-center gap-1.5">
              <Badge variant={isSource ? "accent" : "default"} className="text-[9px]">
                {isSource ? "source" : "target"}
              </Badge>
              <span className="truncate text-[10px] text-muted-foreground">{pkg.category}</span>
              <span className="font-mono text-[10px] text-muted-foreground">v{pkg.version}</span>
            </span>
          </span>
          {pkg.installed && (
            <Badge variant="success" className="shrink-0 gap-1 text-[9px]">
              <CircleCheck className="size-2.5" /> Installed
            </Badge>
          )}
        </div>
        <p className="mt-2 line-clamp-2 text-[11.5px] leading-relaxed text-muted-foreground">
          {pkg.summary}
        </p>
      </button>

      <div className="mt-3 flex items-center gap-2">
        <Button
          size="sm"
          variant={pkg.installed ? "outline" : "default"}
          className="flex-1"
          disabled={installing || pkg.installed}
          onClick={() => actions.installPackage(pkg.name)}
        >
          {pkg.installed ? (
            <>
              <CircleCheck className="size-3.5" /> Installed
            </>
          ) : installing ? (
            <>
              <Loader2 className="size-3.5 animate-spin" /> Installing…
            </>
          ) : (
            <>
              <Download className="size-3.5" /> Install
            </>
          )}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => actions.selectPackage(pkg.name)}
        >
          Details
        </Button>
      </div>
    </div>
  );
}
