import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { reportUiError } from "@/lib/error-reporting";

interface Props {
  viewLabel: string;
  children: ReactNode;
}

interface State {
  failed: boolean;
  error: Error | null;
}

export class ViewErrorBoundary extends Component<Props, State> {
  state: State = {
    failed: false,
    error: null,
  };

  static getDerivedStateFromError(error: Error): State {
    return { failed: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[${this.props.viewLabel}] view crashed`, error, info);
    void reportUiError("view-boundary", error, {
      viewLabel: this.props.viewLabel,
      componentStack: info.componentStack,
    });
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="grid flex-1 place-items-center p-6">
        <div className="w-full max-w-lg rounded-md border border-destructive/30 bg-surface-panel p-5">
          <div className="flex items-start gap-3">
            <span className="grid size-9 shrink-0 place-items-center rounded-md border border-destructive/30 bg-destructive/[0.08]">
              <AlertTriangle className="size-4 text-destructive" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-foreground">
                {this.props.viewLabel} crashed
              </div>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                This view hit an unexpected error. The rest of the app is still available.
              </p>
              <div className="mt-4 flex gap-2">
                <Button
                  size="sm"
                  onClick={() => this.setState({ failed: false, error: null })}
                >
                  <RefreshCw className="size-3.5" /> Reload view
                </Button>
              </div>
              {this.state.error?.message ? (
                <pre className="mt-4 whitespace-pre-wrap rounded-md border border-outline-subtle bg-surface-toolbar p-3 font-mono text-[11px] text-muted-foreground">
                  {this.state.error.message}
                </pre>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    );
  }
}
