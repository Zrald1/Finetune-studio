import React, { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { EmbeddingConfig, DEFAULT_EMBEDDING } from "../types";

interface EmbeddingWidgetProps {
  value: EmbeddingConfig;
  onChange: (config: EmbeddingConfig) => void;
  isEditing: boolean;
}

interface OllamaModel {
  name: string;
}

async function detectOllamaModels(apiUrl: string): Promise<string[]> {
  try {
    const response = await fetch(`${apiUrl}/api/tags`);
    if (!response.ok) return [];
    const data = await response.json();
    return (data.models || []).map((m: OllamaModel) => m.name);
  } catch {
    return [];
  }
}

async function detectLlamaCppModels(apiUrl: string): Promise<string[]> {
  try {
    const response = await fetch(`${apiUrl}/v1/models`);
    if (!response.ok) return [];
    const data = await response.json();
    if (data.data && Array.isArray(data.data)) {
      return data.data.map((m: { id: string }) => m.id);
    }
    return [];
  } catch {
    return [];
  }
}

const EMBEDDING_MODELS: Record<string, string[]> = {
  vllm: ["Qwen/Qwen3-Embedding-8B"],
  ollama: [],
  llamacpp: [],
};

const PROVIDER_LABELS: Record<string, string> = {
  vllm: "vLLM (GPU Server)",
  ollama: "Ollama (Local)",
  llamacpp: "Llama.cpp (Local)",
};

export const EmbeddingWidget: React.FC<EmbeddingWidgetProps> = ({ value, onChange, isEditing }) => {
  const [detecting, setDetecting] = useState(false);
  const [detectedModels, setDetectedModels] = useState<string[]>([]);
  const [lastUrl, setLastUrl] = useState("");

  const detectModels = useCallback(async () => {
    if (!value.apiUrl || value.provider === "vllm") {
      setDetectedModels(EMBEDDING_MODELS.vllm);
      return;
    }

    setDetecting(true);
    try {
      let models: string[] = [];
      if (value.provider === "ollama") {
        models = await detectOllamaModels(value.apiUrl);
      } else if (value.provider === "llamacpp") {
        models = await detectLlamaCppModels(value.apiUrl);
      }
      setDetectedModels(models);
    } catch {
      setDetectedModels([]);
    } finally {
      setDetecting(false);
    }
  }, [value.apiUrl, value.provider]);

  useEffect(() => {
    if (value.apiUrl && value.apiUrl !== lastUrl && value.provider !== "vllm") {
      setLastUrl(value.apiUrl);
      detectModels();
    } else if (value.provider === "vllm") {
      setDetectedModels(EMBEDDING_MODELS.vllm);
    }
  }, [value.apiUrl, value.provider, detectModels, lastUrl]);

  useEffect(() => {
    if (value.provider === "vllm" && !value.modelId) {
      onChange({ ...value, modelId: EMBEDDING_MODELS.vllm[0] });
    }
  }, []);

  const handleProviderChange = (provider: EmbeddingConfig["provider"]) => {
    const defaultUrl = provider === "ollama" ? "http://localhost:11434" :
                       provider === "llamacpp" ? "http://localhost:8080" : "";
    onChange({
      ...DEFAULT_EMBEDDING,
      provider,
      apiUrl: value.apiUrl && provider !== "vllm" ? value.apiUrl : defaultUrl,
      modelId: provider === "vllm" ? EMBEDDING_MODELS.vllm[0] : "",
    });
  };

  const handleManualDetect = async () => {
    if (value.provider === "vllm") return;
    await detectModels();
  };

  const isLocalProvider = value.provider !== "vllm";
  const availableModels = value.provider === "vllm" ? EMBEDDING_MODELS.vllm : detectedModels;

  if (!isEditing) {
    return (
      <div className="widget-block" data-widget="embedding">
        <div className="widget-label">Embedding Provider</div>
        <div className="widget-value">
          <span className="provider-badge">{PROVIDER_LABELS[value.provider]}</span>
        </div>
        {value.apiUrl && (
          <div className="widget-subvalue">
            <span className="sublabel">URL:</span> {value.apiUrl}
          </div>
        )}
        {value.modelId && (
          <div className="widget-subvalue">
            <span className="sublabel">Model:</span> {value.modelId}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="widget-block" data-widget="embedding">
      <div className="widget-label">Embedding Provider</div>

      <div className="provider-selector">
        {(["vllm", "ollama", "llamacpp"] as const).map((provider) => (
          <button
            key={provider}
            className={`provider-btn ${value.provider === provider ? "active" : ""}`}
            onClick={() => handleProviderChange(provider)}
            type="button"
          >
            {PROVIDER_LABELS[provider]}
          </button>
        ))}
      </div>

      {isLocalProvider && (
        <div className="input-group">
          <label className="input-label">API URL</label>
          <div className="url-input-row">
            <input
              type="text"
              className="text-input"
              value={value.apiUrl || ""}
              onChange={(e) => onChange({ ...value, apiUrl: e.target.value })}
              placeholder="http://localhost:11434"
            />
            <button
              className="detect-btn"
              onClick={handleManualDetect}
              disabled={detecting || !value.apiUrl}
              type="button"
            >
              {detecting ? "..." : "Detect"}
            </button>
          </div>
        </div>
      )}

      <div className="input-group">
        <label className="input-label">Embedding Model</label>
        {availableModels.length > 0 ? (
          <select
            className="select-input"
            value={value.modelId || ""}
            onChange={(e) => onChange({ ...value, modelId: e.target.value })}
          >
            <option value="">Select model...</option>
            {availableModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="text-input"
            value={value.modelId || ""}
            onChange={(e) => onChange({ ...value, modelId: e.target.value })}
            placeholder="Enter model ID (e.g., nomic-embed-text)"
          />
        )}
      </div>

      {value.provider === "vllm" && (
        <div className="input-group">
          <label className="input-label">API Key</label>
          <input
            type="password"
            className="text-input"
            value={value.apiKey || ""}
            onChange={(e) => onChange({ ...value, apiKey: e.target.value })}
            placeholder="sk-..."
          />
        </div>
      )}

      {isLocalProvider && detecting && (
        <div className="status-message">Detecting available models...</div>
      )}
      {isLocalProvider && !detecting && isLocalProvider && detectedModels.length === 0 && value.apiUrl && (
        <div className="status-message warning">No embedding models detected. Make sure your local server is running.</div>
      )}
    </div>
  );
};