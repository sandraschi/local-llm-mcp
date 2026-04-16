import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { listModels, type ModelInfo } from "@/api/client";
import { ModelSelector } from "@/components/models/ModelSelector";
import { getDefaults } from "@/api/defaults";

export function Vision() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>();

  useEffect(() => {
    listModels().then((list) => {
      setModels(list);
      const d = getDefaults();
      if (d && list.some((m) => (m.id ?? m.name) === d.model)) {
        setSelectedModel(d.model);
        setSelectedProvider(d.provider);
      } else if (list.length > 0) {
        setSelectedModel(list[0].id ?? list[0].name);
        setSelectedProvider(list[0].provider);
      }
    });
  }, []);

  return (
    <div className="space-y-8 page-enter">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col gap-2">
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <ImageIcon className="h-8 w-8 text-blue-500" />
            Vision Sandbox
          </h2>
          <p className="text-slate-400 max-w-2xl">
            Multimodal intelligence playground. Test image-to-text capabilities with Gemma 4 and local vision-SOTA models.
          </p>
        </div>
        <ModelSelector 
          models={models}
          selectedModel={selectedModel}
          onSelect={(id, provider) => {
            setSelectedModel(id);
            setSelectedProvider(provider);
          }}
        />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="glass-card border-dashed border-2 border-white/10 hover:border-blue-500/50 transition-colors cursor-pointer group">
          <CardContent className="flex flex-col items-center justify-center h-64 space-y-4">
            <div className="p-4 rounded-full bg-blue-500/10 group-hover:bg-blue-500/20 transition-colors">
              <ImageIcon className="h-8 w-8 text-blue-400" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-white">Drop images here</p>
              <p className="text-xs text-slate-500 mt-1">PNG, JPG up to 10MB</p>
            </div>
            <Button variant="outline" className="mt-4 border-white/10 hover:bg-white/5">
              Select Files
            </Button>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
               <Search className="h-4 w-4 text-blue-400" />
               Prompt Analysis
            </CardTitle>
            <CardDescription>
              Instruction for the multimodal engine
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <textarea 
              className="w-full h-32 bg-white/5 border border-white/10 rounded-xl p-4 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all placeholder:text-slate-600"
              placeholder="Describe this image in detail..."
            />
            <Button className="w-full bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_20px_rgba(37,99,235,0.3)]">
              Analyze Scene
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
