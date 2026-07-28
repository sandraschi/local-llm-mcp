import { ExternalLink, Layers } from "lucide-react";
import { APPS_CATALOG } from "@/common/apps-catalog";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function Fleet() {
  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Layers className="h-8 w-8 text-emerald-500" />
          SOTA Fleet Registry
        </h2>
        <p className="text-slate-400 max-w-2xl">
          Unified navigation for the local MCP ecosystem. Launch strategic services and monitor
          cross-service orchestration from this central hub.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {APPS_CATALOG.map((app) => (
          <a
            key={app.id}
            href={app.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group block no-underline transition-all duration-300 transform hover:-translate-y-1"
          >
            <Card className="h-full border-white/5 bg-white/[0.02] hover:bg-white/[0.05] backdrop-blur-xl transition-all border shadow-2xl overflow-hidden group-hover:border-emerald-500/50">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between mb-2">
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20 group-hover:bg-emerald-500/20 transition-colors">
                    <app.icon className="h-5 w-5 text-emerald-400" />
                  </div>
                  <ExternalLink className="h-4 w-4 text-slate-600 group-hover:text-emerald-400 transition-colors" />
                </div>
                <CardTitle className="text-lg text-white group-hover:text-emerald-300 transition-colors">
                  {app.label}
                </CardTitle>
                <CardDescription className="text-slate-400 text-sm line-clamp-2">
                  {app.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mt-2">
                  {app.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wider font-bold bg-white/5 text-slate-500 border border-white/5 shadow-sm"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-500 group-hover:text-emerald-500/70 transition-colors">
                    PORT: {app.port}
                  </span>
                  <span className="text-[10px] font-medium text-emerald-500/50 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">
                    Launch Service
                  </span>
                </div>
              </CardContent>
            </Card>
          </a>
        ))}
      </div>

      <div className="p-8 rounded-3xl border border-dashed border-emerald-500/10 bg-emerald-500/[0.02] flex flex-col items-center justify-center text-center gap-4">
        <div className="w-12 h-12 rounded-full bg-emerald-500/5 border border-emerald-500/10 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
        </div>
        <div>
          <h4 className="text-slate-200 font-medium">Automatic Discovery Active</h4>
          <p className="text-sm text-slate-500">
            Fleet status is continuously monitored via the secondary discovery protocol.
          </p>
        </div>
      </div>
    </div>
  );
}
