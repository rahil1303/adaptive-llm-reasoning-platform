import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000/api";

const MODES = [
  { id: "direct", label: "Direct", desc: "Standard response" },
  { id: "hint_first", label: "Hint-First", desc: "Hints before answer" },
  { id: "guided_reasoning", label: "Guided", desc: "Step-by-step reasoning" },
];

const METRICS = ["cosine", "l2", "dot"];

type Chunk = { text: string; score: number; source: string; chunk_index: number };
type ModelResp = {
  model_id: string;
  model_name: string;
  response: string;
  tokens_used: number;
  latency_ms: number;
  chunks_used: Chunk[];
  mode: string;
};
type Critique = {
  correctness_score: number;
  groundedness_score: number;
  completeness_score: number;
  issues: string[];
  suggestions: string[];
  summary: string;
};
type ChatEntry = {
  query: string;
  responses: ModelResp[];
  session_id: string;
  critiques?: Record<string, Critique>;
};

function ScoreBar({ label, score }: { label: string; score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "#34d399" : pct >= 40 ? "#fbbf24" : "#f87171";
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 2 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "var(--bg-tertiary)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

function ChunkInspector({ chunks }: { chunks: Chunk[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!chunks.length) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          background: "none", border: "1px solid var(--border)", borderRadius: 4,
          padding: "4px 10px", fontSize: 12, cursor: "pointer", color: "var(--text-secondary)"
        }}
      >
        {expanded ? "▼" : "▶"} {chunks.length} chunks retrieved
      </button>
      {expanded && (
        <div style={{ marginTop: 8 }}>
          {chunks.map((c, i) => (
            <div key={i} style={{
              padding: "8px 10px", marginBottom: 6, background: "var(--bg-secondary)",
              borderRadius: 6, borderLeft: `3px solid ${c.score > 0.8 ? "#34d399" : c.score > 0.5 ? "#fbbf24" : "#94a3b8"}`,
              fontSize: 13,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 11, color: "var(--text-secondary)" }}>
                <span>{c.source} — chunk #{c.chunk_index}</span>
                <span style={{ fontWeight: 600 }}>score: {c.score.toFixed(4)}</span>
              </div>
              <div style={{ lineHeight: 1.5 }}>{c.text.slice(0, 300)}{c.text.length > 300 ? "…" : ""}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ResponseCard({ resp, critique, onCritique, loading }: {
  resp: ModelResp; critique?: Critique; onCritique: () => void; loading: boolean;
}) {
  return (
    <div style={{
      flex: 1, minWidth: 320, background: "var(--bg-secondary)", borderRadius: 10,
      padding: 16, border: "1px solid var(--border)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 14 }}>{resp.model_name}</span>
          <span style={{ fontSize: 11, color: "var(--text-secondary)", marginLeft: 8 }}>
            {resp.tokens_used} tok · {resp.latency_ms.toFixed(0)}ms · {resp.mode}
          </span>
        </div>
        <button
          onClick={onCritique}
          disabled={loading || !!critique}
          style={{
            background: critique ? "var(--bg-tertiary)" : "var(--accent)",
            color: critique ? "var(--text-secondary)" : "#fff",
            border: "none", borderRadius: 5, padding: "4px 12px",
            fontSize: 12, cursor: critique ? "default" : "pointer",
          }}
        >
          {loading ? "…" : critique ? "Critiqued" : "Critique"}
        </button>
      </div>

      <div style={{ fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
        {resp.response}
      </div>

      {critique && (
        <div style={{ marginTop: 14, padding: 12, background: "var(--bg-primary)", borderRadius: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Critique Scores</div>
          <ScoreBar label="Correctness" score={critique.correctness_score} />
          <ScoreBar label="Groundedness" score={critique.groundedness_score} />
          <ScoreBar label="Completeness" score={critique.completeness_score} />
          {critique.issues.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 12 }}>
              <span style={{ fontWeight: 600, color: "#f87171" }}>Issues:</span>{" "}
              {critique.issues.join("; ")}
            </div>
          )}
          {critique.summary && (
            <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-secondary)" }}>{critique.summary}</div>
          )}
        </div>
      )}

      <ChunkInspector chunks={resp.chunks_used} />
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [critiquing, setCritiquing] = useState<Record<string, boolean>>({});

  // Settings
  const [selectedModels, setSelectedModels] = useState<string[]>(["llama-3.3-70b"]);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [mode, setMode] = useState("direct");
  const [useRag, setUseRag] = useState(true);
  const [topK, setTopK] = useState(5);
  const [metric, setMetric] = useState("cosine");
  const [docStatus, setDocStatus] = useState<any>(null);
  const [uploading, setUploading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API}/models/list`).then(r => r.json()).then(d => setAvailableModels(d.models)).catch(() => {});
    fetch(`${API}/documents/status`).then(r => r.json()).then(d => setDocStatus(d)).catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const handleQuery = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    try {
      const resp = await fetch(`${API}/chat/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          model_ids: selectedModels,
          mode,
          use_rag: useRag,
          top_k: topK,
          similarity_metric: metric,
        }),
      });
      const data = await resp.json();
      setHistory(prev => [...prev, data]);
      setQuery("");
    } catch (e: any) {
      alert("Error: " + e.message);
    }
    setLoading(false);
  };

  const handleCritique = async (entryIdx: number, resp: ModelResp) => {
    const key = `${entryIdx}-${resp.model_id}`;
    setCritiquing(prev => ({ ...prev, [key]: true }));
    try {
      const r = await fetch(`${API}/critique/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: history[entryIdx].query,
          response_text: resp.response,
          context_chunks: resp.chunks_used.map(c => c.text),
        }),
      });
      const critique = await r.json();
      setHistory(prev => {
        const updated = [...prev];
        if (!updated[entryIdx].critiques) updated[entryIdx].critiques = {};
        updated[entryIdx].critiques![resp.model_id] = critique;
        return updated;
      });
    } catch (e) {
      console.error(e);
    }
    setCritiquing(prev => ({ ...prev, [key]: false }));
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await fetch(`${API}/documents/upload`, { method: "POST", body: form });
      const data = await r.json();
      setDocStatus(await (await fetch(`${API}/documents/status`)).json());
      alert(`Ingested ${data.chunk_count} chunks from ${data.filename}`);
    } catch (e: any) {
      alert("Upload failed: " + e.message);
    }
    setUploading(false);
  };

  const toggleModel = (id: string) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg-primary)",
      color: "var(--text-primary)",
      fontFamily: "'DM Sans', system-ui, sans-serif",
    }}>
      {/* Header */}
      <header style={{
        padding: "14px 24px",
        borderBottom: "1px solid var(--border)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#34d399" }} />
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.02em" }}>
            AI Reasoning Platform
          </span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {docStatus && (
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
              {docStatus.total_chunks} chunks indexed
            </span>
          )}
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            style={{
              background: "var(--bg-secondary)", border: "1px solid var(--border)",
              borderRadius: 6, padding: "6px 14px", fontSize: 13, cursor: "pointer",
              color: "var(--text-primary)",
            }}
          >
            ⚙ Settings
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      {settingsOpen && (
        <div style={{
          padding: "16px 24px", borderBottom: "1px solid var(--border)",
          background: "var(--bg-secondary)", display: "flex", flexWrap: "wrap", gap: 20,
        }}>
          {/* Models */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Models</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {availableModels.map(m => (
                <button
                  key={m.id}
                  onClick={() => toggleModel(m.id)}
                  style={{
                    padding: "4px 10px", borderRadius: 5, fontSize: 12, cursor: "pointer",
                    border: "1px solid var(--border)",
                    background: selectedModels.includes(m.id) ? "var(--accent)" : "var(--bg-primary)",
                    color: selectedModels.includes(m.id) ? "#fff" : "var(--text-primary)",
                    opacity: m.has_api_key ? 1 : 0.4,
                  }}
                  disabled={!m.has_api_key}
                  title={m.has_api_key ? m.display_name : `No API key set (${m.display_name})`}
                >
                  {m.display_name}
                </button>
              ))}
            </div>
          </div>

          {/* Mode */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Interaction Mode</div>
            <div style={{ display: "flex", gap: 6 }}>
              {MODES.map(m => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  style={{
                    padding: "4px 10px", borderRadius: 5, fontSize: 12, cursor: "pointer",
                    border: "1px solid var(--border)",
                    background: mode === m.id ? "var(--accent)" : "var(--bg-primary)",
                    color: mode === m.id ? "#fff" : "var(--text-primary)",
                  }}
                  title={m.desc}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* RAG Settings */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Retrieval</div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
                <input type="checkbox" checked={useRag} onChange={e => setUseRag(e.target.checked)} />
                RAG
              </label>
              <select
                value={metric}
                onChange={e => setMetric(e.target.value)}
                style={{ fontSize: 12, padding: "3px 6px", borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg-primary)", color: "var(--text-primary)" }}
              >
                {METRICS.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <label style={{ fontSize: 12 }}>
                Top-K:
                <input
                  type="number" min={1} max={20} value={topK}
                  onChange={e => setTopK(Number(e.target.value))}
                  style={{ width: 40, marginLeft: 4, fontSize: 12, padding: "3px", borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg-primary)", color: "var(--text-primary)" }}
                />
              </label>
            </div>
          </div>

          {/* Upload */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Documents</div>
            <label style={{
              padding: "6px 14px", borderRadius: 5, fontSize: 12, cursor: "pointer",
              border: "1px dashed var(--border)", display: "inline-block",
            }}>
              {uploading ? "Uploading…" : "Upload Document"}
              <input
                type="file"
                accept=".pdf,.txt,.md"
                style={{ display: "none" }}
                onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])}
                disabled={uploading}
              />
            </label>
          </div>
        </div>
      )}

      {/* Chat Area */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "20px 24px", paddingBottom: 120 }}>
        {history.length === 0 && (
          <div style={{ textAlign: "center", marginTop: 80, color: "var(--text-secondary)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>⚡</div>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 6 }}>Ask anything</div>
            <div style={{ fontSize: 14 }}>
              Upload a document and query it across multiple models.
              <br />Compare responses, inspect retrieval, and run critiques.
            </div>
          </div>
        )}

        {history.map((entry, idx) => (
          <div key={idx} style={{ marginBottom: 32 }}>
            {/* Query */}
            <div style={{
              background: "var(--accent)", color: "#fff", padding: "10px 16px",
              borderRadius: "10px 10px 2px 10px", display: "inline-block",
              maxWidth: "70%", marginBottom: 14, fontWeight: 500, fontSize: 14,
            }}>
              {entry.query}
            </div>

            {/* Responses side by side */}
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
              {entry.responses.map(resp => (
                <ResponseCard
                  key={resp.model_id}
                  resp={resp}
                  critique={entry.critiques?.[resp.model_id]}
                  onCritique={() => handleCritique(idx, resp)}
                  loading={critiquing[`${idx}-${resp.model_id}`] || false}
                />
              ))}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Input Bar */}
      <div style={{
        position: "fixed", bottom: 0, left: 0, right: 0,
        padding: "12px 24px", background: "var(--bg-primary)",
        borderTop: "1px solid var(--border)",
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", gap: 10 }}>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleQuery()}
            placeholder="Ask a question…"
            style={{
              flex: 1, padding: "12px 16px", borderRadius: 8,
              border: "1px solid var(--border)", background: "var(--bg-secondary)",
              color: "var(--text-primary)", fontSize: 14, outline: "none",
            }}
            disabled={loading}
          />
          <button
            onClick={handleQuery}
            disabled={loading || !query.trim()}
            style={{
              padding: "12px 24px", borderRadius: 8, border: "none",
              background: "var(--accent)", color: "#fff", fontSize: 14,
              fontWeight: 600, cursor: loading ? "wait" : "pointer",
              opacity: loading || !query.trim() ? 0.5 : 1,
            }}
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
      </div>

      <style>{`
        :root {
          --bg-primary: #0f1117;
          --bg-secondary: #1a1d27;
          --bg-tertiary: #252836;
          --text-primary: #e4e6eb;
          --text-secondary: #8b8fa3;
          --border: #2a2d3a;
          --accent: #6366f1;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg-primary); }
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
      `}</style>
    </div>
  );
}
