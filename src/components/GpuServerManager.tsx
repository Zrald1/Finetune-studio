import { useEffect, useMemo, useState } from "react";
import type {
  AppConfig,
  DigitalOceanAccount,
  DigitalOceanDroplet,
  DigitalOceanImage,
  DigitalOceanProject,
  DigitalOceanSize,
  DigitalOceanSshKey,
} from "../types";
import { DEFAULT_DIGITAL_OCEAN } from "../types";
import { api } from "../lib/tauri";
import { CheckCircle2, Cloud, Cpu, Loader2, RefreshCw, Server, Trash2, Zap } from "lucide-react";

interface ImageOption {
  value: string;
  label: string;
}

interface Props {
  config: AppConfig;
  onConfigChange: (patch: Partial<AppConfig>) => void;
}

const QUICK_START_IMAGES: ImageOption[] = [
  { value: "220895104", label: "ROCm Software | ROCm 7.2" },
  { value: "201932560", label: "OpenAI GPT OSS | ROCm 7.0, vLLM 0.9.2" },
  { value: "221160341", label: "vLLM | vLLM 0.17.1, ROCm 7.2.0" },
  { value: "221157360", label: "SGLang | SGLang 0.5.9, ROCm 7.0.0" },
  { value: "201616009", label: "PyTorch | PyTorch 2.6.0, ROCm 7.0.0" },
  { value: "201813315", label: "Megatron | Megatron-LM 0.10.0, ROCm 7.0" },
  { value: "194144121", label: "JAX | JAX 0.4.35, ROCm 6.4.2" },
];

function publicIp(droplet: DigitalOceanDroplet) {
  return droplet.networks?.v4?.find((addr) => addr.type === "public")?.ipAddress || "";
}

function imageValue(image: DigitalOceanImage) {
  return image.slug || String(image.id);
}

function imageLabel(image: DigitalOceanImage) {
  const version = image.description || image.distribution || "";
  return `${image.name}${version ? ` | ${version}` : ""}`;
}

function quickStartImage(value: string) {
  return QUICK_START_IMAGES.find((image) => image.value === value);
}

function isAmdGpuSlug(slug: string) {
  return /^gpu-mi\d+x?/i.test(slug.trim());
}

function gpuModel(size: DigitalOceanSize) {
  const slugModel = size.slug.match(/gpu-(mi\d+x?)/i)?.[1]?.toUpperCase();
  return size.gpuInfo?.model || slugModel || "Unlabeled AMD GPU";
}

function customGpuSize(slug: string): DigitalOceanSize | undefined {
  const clean = slug.trim();
  if (!isAmdGpuSlug(clean)) return undefined;
  return {
    slug: clean,
    memory: 0,
    vcpus: 0,
    disk: 0,
    transfer: 0,
    priceMonthly: null,
    priceHourly: null,
    regions: [],
    available: true,
    description: "Contract GPU",
    gpuInfo: {
      model: clean.match(/gpu-(mi\d+x?)/i)?.[1]?.toUpperCase(),
    },
  };
}

export default function GpuServerManager({ config, onConfigChange }: Props) {
  const digitalOcean = { ...DEFAULT_DIGITAL_OCEAN, ...(config.digitalOcean ?? {}) };
  const [sizes, setSizes] = useState<DigitalOceanSize[]>([]);
  const [images, setImages] = useState<DigitalOceanImage[]>([]);
  const [sshKeys, setSshKeys] = useState<DigitalOceanSshKey[]>([]);
  const [projects, setProjects] = useState<DigitalOceanProject[]>([]);
  const [droplets, setDroplets] = useState<DigitalOceanDroplet[]>([]);
  const [account, setAccount] = useState<DigitalOceanAccount | null>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const visibleImages = useMemo(() => {
    const seen = new Set<string>();
    const options: ImageOption[] = [];
    for (const image of QUICK_START_IMAGES) {
      seen.add(image.value);
      options.push(image);
    }
    for (const image of images) {
      const value = imageValue(image);
      if (seen.has(value)) continue;
      seen.add(value);
      options.push({ value, label: imageLabel(image) });
    }
    return options;
  }, [images]);

  const sizesByModel = useMemo(() => {
    const groups = new Map<string, DigitalOceanSize[]>();
    for (const size of sizes) {
      const model = gpuModel(size);
      groups.set(model, [...(groups.get(model) || []), size]);
    }
    return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [sizes]);

  const gpuDroplets = droplets;

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === digitalOcean.projectId),
    [digitalOcean.projectId, projects],
  );

  const selectedSshKeys = useMemo(
    () => digitalOcean.sshKeys.split(",").map((s) => s.trim()).filter(Boolean),
    [digitalOcean.sshKeys],
  );

  const patchDigitalOcean = (patch: Partial<typeof digitalOcean>) => {
    onConfigChange({ digitalOcean: { ...digitalOcean, ...patch } });
  };

  const sync = async () => {
    if (!digitalOcean.apiKey.trim()) {
      setMessage("Add the DigitalOcean API token in Credentials first.");
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const [nextAccount, nextSizes, nextImages, nextKeys, nextProjects, nextDroplets] = await Promise.all([
        api.doGetAccount(digitalOcean),
        api.doListGpuSizes(digitalOcean),
        api.doListImages(digitalOcean),
        api.doListSshKeys(digitalOcean),
        api.doListProjects(digitalOcean),
        api.doListGpuDroplets(digitalOcean),
      ]);
      setAccount(nextAccount);
      setSizes(nextSizes);
      setImages(nextImages);
      setSshKeys(nextKeys);
      setProjects(nextProjects);
      setDroplets(nextDroplets);

      const currentCustomSize = customGpuSize(digitalOcean.size);
      const nextSize = nextSizes.find((size) => size.slug === digitalOcean.size) || currentCustomSize || nextSizes[0];
      const nextImage =
        nextImages.find((image) => imageValue(image) === digitalOcean.image) ||
        quickStartImage(digitalOcean.image) ||
        nextImages[0];
      const nextProject = nextProjects.find((project) => project.id === digitalOcean.projectId)
        ? digitalOcean.projectId
        : nextProjects.find((project) => project.isDefault)?.id || nextProjects[0]?.id || "";
      patchDigitalOcean({
        size: nextSize?.slug || digitalOcean.size,
        region: digitalOcean.region,
        image: nextImage ? ("value" in nextImage ? nextImage.value : imageValue(nextImage)) : digitalOcean.image,
        sshKeys: digitalOcean.sshKeys || nextKeys[0]?.id?.toString() || "",
        projectId: nextProject,
      });
      setMessage(`Loaded DigitalOcean data: ${nextSizes.length} AMD GPU plans, ${nextImages.length} ROCm images, ${nextKeys.length} SSH keys, and ${nextDroplets.length} GPU droplets.`);
    } catch (e: any) {
      setMessage(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (digitalOcean.apiKey && sizes.length === 0 && !loading) {
      sync();
    }
  }, [digitalOcean.apiKey]);

  const toggleSshKey = (id: string) => {
    const current = new Set(selectedSshKeys);
    if (current.has(id)) current.delete(id);
    else if (current.size < 4) current.add(id);
    patchDigitalOcean({ sshKeys: Array.from(current).join(",") });
  };

  const create = async () => {
    setCreating(true);
    setMessage(null);
    try {
      const droplet = await api.doCreateGpuDroplet(digitalOcean);
      setMessage(`Created ${droplet.name}. It can take a few minutes before the public IP is assigned.`);
      await sync();
    } catch (e: any) {
      setMessage(String(e));
    } finally {
      setCreating(false);
    }
  };

  const destroy = async (droplet: DigitalOceanDroplet) => {
    if (!window.confirm(`Destroy "${droplet.name}" (#${droplet.id})? This is permanent.`)) return;
    setLoading(true);
    setMessage(null);
    try {
      await api.doDestroyDroplet(digitalOcean, droplet.id);
      setMessage(`Destroy requested for ${droplet.name}.`);
      await sync();
    } catch (e: any) {
      setMessage(String(e));
    } finally {
      setLoading(false);
    }
  };

  const canCreate = !!digitalOcean.apiKey && !!digitalOcean.dropletName && !!digitalOcean.size && !!digitalOcean.image && selectedSshKeys.length > 0;

  return (
    <div className="premium-card rounded-2xl animate-premium overflow-hidden">
      <div className="px-8 py-6 border-b theme-surface-soft bg-white/[0.02] flex items-start justify-between gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.3em] theme-accent font-black font-mono">DigitalOcean GPU Control</p>
          <h2 className="text-2xl-fluid font-serif italic text-white tracking-tight font-black mt-1">GPU Servers</h2>
          <p className="text-sm-fluid theme-muted font-medium opacity-80 mt-2">Plans, images, SSH keys, and projects are loaded from your DigitalOcean account.</p>
          {account && (
            <div className="mt-3 inline-flex items-center gap-2 rounded-full border border-white/5 bg-white/[0.03] px-3 py-1.5 text-[10px] uppercase tracking-widest font-black theme-muted">
              <Cloud className="w-3.5 h-3.5 theme-accent" />
              {account.team?.name ? `Team: ${account.team.name}` : `Account: ${account.name || account.email || account.uuid}`}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={sync}
          disabled={loading || !digitalOcean.apiKey}
          className="px-5 py-2.5 rounded-xl border border-white/10 theme-surface-soft theme-text hover:border-theme-accent/40 text-[10px] uppercase tracking-widest font-black disabled:opacity-30 flex items-center gap-3"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Sync Account
        </button>
      </div>

      <div className="p-8 space-y-8">
        {!digitalOcean.apiKey && (
          <div className="p-5 rounded-2xl border border-amber-500/20 bg-amber-500/5 text-amber-300 text-sm-fluid font-mono">
            Add your DigitalOcean API token in Credentials before managing GPU servers.
          </div>
        )}

        <div className="space-y-5">
          {sizesByModel.length === 0 ? (
            <div className="p-5 rounded-2xl border border-white/5 bg-white/[0.015] theme-muted font-mono text-sm-fluid">
              Sync returned no AMD GPU plans for this DigitalOcean token.
            </div>
          ) : sizesByModel.map(([model, modelSizes]) => (
            <div key={model} className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="text-[10px] uppercase tracking-[0.25em] theme-accent font-black font-mono">{model}</span>
                <span className="h-px flex-1 bg-white/5" />
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                {modelSizes.map((size) => (
                  <button
                    key={size.slug}
                    type="button"
                    onClick={() => {
                      patchDigitalOcean({ size: size.slug });
                    }}
                    className={`text-left rounded-2xl border p-5 transition-all premium-button ${digitalOcean.size === size.slug ? "theme-accent-soft theme-accent border-theme-accent/40" : "border-white/5 bg-white/[0.015] theme-muted hover:theme-text"}`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-xs-fluid uppercase tracking-[0.18em] font-black">{size.slug}</span>
                      {digitalOcean.size === size.slug && <CheckCircle2 className="w-4 h-4" />}
                    </div>
                    <div className="mt-4 text-2xl font-black text-white font-mono">{size.gpuInfo?.count || 0} GPU</div>
                    <div className="mt-2 text-[10px] uppercase tracking-tight font-mono theme-muted">
                      {size.memory / 1024} GB RAM | {size.vcpus} vCPU | {size.disk} GB disk
                    </div>
                    <div className="mt-3 text-[11px] theme-accent font-black font-mono">
                      {typeof size.priceHourly === "number" ? `$${size.priceHourly.toFixed(2)}/hr` : "Contract pricing"}
                    </div>
                    <div className="mt-2 text-[10px] font-mono theme-muted truncate" title={size.regions.length ? size.regions.join(", ") : "DigitalOcean may omit AMD GPU regions from the API; set Region Slug to atl1 if the dashboard offers ATL1."}>
                      {size.regions.length
                        ? `Regions: ${size.regions.join(", ")}`
                        : "No regions reported — try atl1"}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-3 pb-3 border-b border-white/5">
              <Cpu className="w-5 h-5 theme-accent" />
              <div>
                <h3 className="text-base-fluid text-white font-black">Create Server</h3>
                <p className="text-[10px] uppercase tracking-widest theme-muted font-mono mt-1">
                  Target: {account?.team?.name ? account.team.name : account ? "Personal account" : "Sync account"}{selectedProject ? ` | ${selectedProject.name}` : ""}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2 sm:col-span-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">GPU Size Slug</label>
                <input value={digitalOcean.size} onChange={(e) => patchDigitalOcean({ size: e.target.value })} placeholder="gpu-mi...-contracted" className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none" />
              </div>
              <div className="space-y-2 sm:col-span-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Region Slug</label>
                <input value={digitalOcean.region} onChange={(e) => patchDigitalOcean({ region: e.target.value })} placeholder="atl1" className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none" />
              </div>
              <div className="space-y-2 sm:col-span-2">
                <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Name</label>
                <input value={digitalOcean.dropletName} onChange={(e) => patchDigitalOcean({ dropletName: e.target.value })} placeholder="Enter a DigitalOcean droplet name" className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono focus:outline-none" />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Image</label>
              <select value={digitalOcean.image} onChange={(e) => patchDigitalOcean({ image: e.target.value })} className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono bg-black/20 text-white focus:outline-none">
                {visibleImages.map((image) => <option key={image.value} value={image.value}>{image.label}</option>)}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">Project</label>
              <select value={digitalOcean.projectId} onChange={(e) => patchDigitalOcean({ projectId: e.target.value })} className="w-full px-4 py-3 premium-input rounded-xl text-sm-fluid font-mono bg-black/20 text-white focus:outline-none">
                <option value="">Default assignment</option>
                {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
              </select>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] uppercase tracking-widest theme-muted font-black ml-1">SSH Keys</label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {sshKeys.map((key) => {
                  const id = String(key.id);
                  return (
                  <button key={id} type="button" onClick={() => toggleSshKey(id)} className={`px-4 py-3 rounded-xl border text-left text-[10px] uppercase tracking-widest font-black ${selectedSshKeys.includes(id) ? "theme-accent-soft theme-accent border-theme-accent/40" : "border-white/5 bg-white/[0.015] theme-muted"}`}>
                    <span className="block text-white truncate">{key.name || key.fingerprint || `SSH Key ${id}`}</span>
                    <span className="block mt-1 theme-muted font-mono opacity-70">ID {id}</span>
                  </button>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {(["monitoring", "ipv6", "backups"] as const).map((key) => (
                <label key={key} className="flex items-center gap-3 px-4 py-3 rounded-xl border border-white/5 bg-white/[0.015] cursor-pointer">
                  <input type="checkbox" checked={!!digitalOcean[key]} onChange={(e) => patchDigitalOcean({ [key]: e.target.checked })} className="h-4 w-4 accent-[rgb(var(--app-accent-rgb))]" />
                  <span className="text-[10px] uppercase tracking-widest font-black theme-muted">{key}</span>
                </label>
              ))}
            </div>

            <button type="button" onClick={create} disabled={!canCreate || creating || loading} className="w-full theme-accent-bg text-black py-4 rounded-2xl font-black text-sm-fluid uppercase tracking-[0.25em] premium-button disabled:opacity-30 flex items-center justify-center gap-3">
              {creating ? <Loader2 className="w-5 h-5 animate-spin text-black" /> : <Zap className="w-5 h-5 text-black" />}
              Create GPU Server
            </button>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-3 pb-3 border-b border-white/5">
              <Server className="w-5 h-5 theme-accent" />
              <h3 className="text-base-fluid text-white font-black">Active Droplets</h3>
            </div>
            <div className="rounded-2xl border border-white/5 overflow-hidden bg-black/10">
              {gpuDroplets.length === 0 ? (
                <div className="p-10 text-center theme-muted font-serif italic">No AMD GPU droplets loaded.</div>
              ) : gpuDroplets.map((droplet) => {
                const ip = publicIp(droplet);
                return (
                  <div key={droplet.id} className="p-4 border-b border-white/5 last:border-b-0 flex flex-col sm:flex-row gap-4 sm:items-center justify-between">
                    <div>
                      <div className="text-white font-black">{droplet.name}</div>
                      <div className="text-[10px] theme-muted font-mono uppercase mt-1">#{droplet.id} | {droplet.status} | {droplet.sizeSlug || "unknown"} | {ip || "IP pending"}</div>
                    </div>
                    <div className="flex gap-2">
                      <button type="button" disabled={!ip} onClick={() => onConfigChange({ ssh: { ...config.ssh, host: ip, username: config.ssh.username || "root" } })} className="px-3 py-2 rounded-lg border border-white/10 theme-surface-soft theme-text text-[9px] uppercase tracking-widest font-black disabled:opacity-30">Use IP</button>
                      <button type="button" onClick={() => destroy(droplet)} disabled={loading} className="px-3 py-2 rounded-lg border border-red-500/20 bg-red-500/10 text-red-400 text-[9px] uppercase tracking-widest font-black disabled:opacity-30 flex items-center gap-2"><Trash2 className="w-3.5 h-3.5" />Destroy</button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {message && (
          <div className={`p-4 rounded-2xl border text-sm-fluid font-mono ${message.toLowerCase().includes("failed") || message.toLowerCase().includes("error") ? "bg-red-500/5 border-red-500/30 text-red-300" : "bg-emerald-500/5 border-emerald-500/30 text-emerald-300"}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}
