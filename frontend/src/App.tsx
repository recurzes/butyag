import { useState, useRef, useCallback, type DragEvent, type ChangeEvent } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "";

// ── Color tokens ──────────────────────────────────────────────────────────────
// Deep clinical navy + clean white + alert red/green — no warm cream, no acid green
// Signature: the scan-line animation on the upload zone that pulses like a real X-ray reader
const COLORS = {
  navy:       "#0A1628",
  navyMid:    "#122040",
  navyLight:  "#1C3058",
  slate:      "#8FA3BF",
  white:      "#F0F4F8",
  pneumonia:  "#E53E3E",
  normal:     "#38A169",
  accent:     "#4A9EFF",
} as const;

type UIState = "idle" | "loading" | "success" | "error";

interface PredictionResult {
  prediction: "PNEUMONIA" | "NORMAL";
  confidence: number;
  pneumonia_prob: number;
  normal_prob: number;
  gradcam_base64?: string;
  threshold: number | string;
  error?: string;
}

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: ${COLORS.navy};
    color: ${COLORS.white};
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
  }

  .app {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }

  /* ── Header ── */
  .header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 48px;
    border-bottom: 1px solid ${COLORS.navyLight};
    padding-bottom: 24px;
  }
  .header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: ${COLORS.white};
  }
  .header-title span { color: ${COLORS.accent}; }
  .header-sub {
    font-size: 13px;
    color: ${COLORS.slate};
    font-weight: 300;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .header-badge {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: ${COLORS.accent};
    border: 1px solid ${COLORS.navyLight};
    padding: 4px 10px;
    border-radius: 4px;
  }

  /* ── Layout ── */
  .layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 700px) {
    .layout { grid-template-columns: 1fr; }
  }

  /* ── Upload panel ── */
  .upload-panel {
    background: ${COLORS.navyMid};
    border: 1px solid ${COLORS.navyLight};
    border-radius: 12px;
    padding: 28px;
  }
  .panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: ${COLORS.slate};
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  /* ── Drop zone ── */
  .dropzone {
    border: 2px dashed ${COLORS.navyLight};
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: ${COLORS.accent};
    background: rgba(74, 158, 255, 0.04);
  }
  .dropzone.has-image {
    border-style: solid;
    border-color: ${COLORS.navyLight};
    padding: 0;
  }

  /* Scan-line animation */
  @keyframes scanline {
    0%   { transform: translateY(-100%); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateY(400%); opacity: 0; }
  }
  .scanline {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, ${COLORS.accent}, transparent);
    animation: scanline 2.5s ease-in-out infinite;
    pointer-events: none;
    top: 0;
  }

  .dropzone-icon {
    width: 40px;
    height: 40px;
    color: ${COLORS.slate};
  }
  .dropzone-text {
    font-size: 14px;
    color: ${COLORS.slate};
    line-height: 1.6;
  }
  .dropzone-text strong {
    color: ${COLORS.accent};
    cursor: pointer;
  }
  .dropzone-hint {
    font-size: 11px;
    color: ${COLORS.navyLight};
    font-family: 'JetBrains Mono', monospace;
  }

  .preview-img {
    width: 100%;
    border-radius: 6px;
    display: block;
    max-height: 300px;
    object-fit: contain;
    background: #000;
  }

  /* ── Error message ── */
  .error-msg {
    margin-top: 10px;
    font-size: 12px;
    color: ${COLORS.pneumonia};
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Analyze button ── */
  .analyze-btn {
    margin-top: 16px;
    width: 100%;
    padding: 14px;
    background: ${COLORS.accent};
    color: ${COLORS.navy};
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.3px;
    transition: opacity 0.15s, transform 0.1s;
  }
  .analyze-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
  .analyze-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }

  /* ── Result panel ── */
  .result-panel {
    background: ${COLORS.navyMid};
    border: 1px solid ${COLORS.navyLight};
    border-radius: 12px;
    padding: 28px;
    min-height: 340px;
    display: flex;
    flex-direction: column;
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: ${COLORS.navyLight};
  }
  .empty-state p {
    font-size: 13px;
    color: ${COLORS.slate};
    opacity: 0.5;
    text-align: center;
  }

  /* ── Loading ── */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid ${COLORS.navyLight};
    border-top-color: ${COLORS.accent};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: auto;
  }
  .loading-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
  }
  .loading-state p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: ${COLORS.slate};
  }

  /* ── Prediction badge ── */
  .prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 6px;
    margin-bottom: 20px;
  }
  .prediction-badge.pneumonia {
    background: rgba(229, 62, 62, 0.12);
    border: 1px solid rgba(229, 62, 62, 0.3);
  }
  .prediction-badge.normal {
    background: rgba(56, 161, 105, 0.12);
    border: 1px solid rgba(56, 161, 105, 0.3);
  }
  .badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
  }
  .badge-dot.pneumonia { background: ${COLORS.pneumonia}; }
  .badge-dot.normal    { background: ${COLORS.normal}; }
  .badge-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 1px;
  }
  .badge-label.pneumonia { color: ${COLORS.pneumonia}; }
  .badge-label.normal    { color: ${COLORS.normal}; }

  /* ── Metric rows ── */
  .metric-row {
    margin-bottom: 14px;
  }
  .metric-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
  }
  .metric-name {
    font-size: 12px;
    color: ${COLORS.slate};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-value {
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: ${COLORS.white};
  }
  .bar-track {
    height: 5px;
    background: ${COLORS.navyLight};
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
  }

  /* ── Grad-CAM section ── */
  .gradcam-section {
    margin-top: 20px;
    border-top: 1px solid ${COLORS.navyLight};
    padding-top: 20px;
  }
  .gradcam-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 10px;
  }
  .gradcam-img-wrap {
    border-radius: 6px;
    overflow: hidden;
    background: #000;
  }
  .gradcam-img-wrap img {
    width: 100%;
    display: block;
  }
  .gradcam-caption {
    font-size: 10px;
    color: ${COLORS.slate};
    text-align: center;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Threshold note ── */
  .threshold-note {
    margin-top: 14px;
    font-size: 11px;
    color: ${COLORS.slate};
    font-family: 'JetBrains Mono', monospace;
    opacity: 0.7;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    margin-top: 40px;
    padding: 14px 18px;
    border: 1px solid ${COLORS.navyLight};
    border-radius: 8px;
    font-size: 12px;
    color: ${COLORS.slate};
    line-height: 1.6;
    opacity: 0.7;
  }
  .disclaimer strong { color: ${COLORS.white}; opacity: 1; }
`;

// ── Upload Icon ───────────────────────────────────────────────────────────────
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
         strokeLinecap="round" strokeLinejoin="round" className="dropzone-icon">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}

// ── Confidence Bar ────────────────────────────────────────────────────────────
interface MetricBarProps {
  label: string;
  value: number;
  color: string;
}

function MetricBar({ label, value, color }: MetricBarProps) {
  return (
    <div className="metric-row">
      <div className="metric-header">
        <span className="metric-name">{label}</span>
        <span className="metric-value">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
    </div>
  );
}

// ── Result Panel ──────────────────────────────────────────────────────────────
interface ResultPanelProps {
  state: UIState;
  result: PredictionResult | null;
  previewUrl: string | null;
}

function ResultPanel({ state, result, previewUrl }: ResultPanelProps) {
  if (state === "idle") {
    return (
      <div className="result-panel">
        <div className="panel-label">Result</div>
        <div className="empty-state">
          <p>Upload a chest X-ray and click<br />"Analyze X-Ray" to see results.</p>
        </div>
      </div>
    );
  }

  if (state === "loading") {
    return (
      <div className="result-panel">
        <div className="panel-label">Result</div>
        <div className="loading-state">
          <div className="spinner" />
          <p>Running inference...</p>
        </div>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="result-panel">
        <div className="panel-label">Result</div>
        <div className="empty-state">
          <p style={{ color: COLORS.pneumonia }}>
            {result?.error || "Analysis failed. Please try again."}
          </p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const isPneumonia = result.prediction === "PNEUMONIA";
  const cls = isPneumonia ? "pneumonia" : "normal";
  const primaryColor = isPneumonia ? COLORS.pneumonia : COLORS.normal;

  return (
    <div className="result-panel">
      <div className="panel-label">Result</div>

      <div className={`prediction-badge ${cls}`}>
        <div className={`badge-dot ${cls}`} />
        <span className={`badge-label ${cls}`}>{result.prediction}</span>
      </div>

      <MetricBar
        label="Confidence"
        value={result.confidence}
        color={primaryColor}
      />
      <MetricBar
        label="Pneumonia prob."
        value={result.pneumonia_prob}
        color={COLORS.pneumonia}
      />
      <MetricBar
        label="Normal prob."
        value={result.normal_prob}
        color={COLORS.normal}
      />

      {result.gradcam_base64 && previewUrl && (
        <div className="gradcam-section">
          <div className="panel-label">Grad-CAM Explainability</div>
          <div className="gradcam-grid">
            <div>
              <div className="gradcam-img-wrap">
                <img src={previewUrl} alt="Original X-Ray" />
              </div>
              <div className="gradcam-caption">Original</div>
            </div>
            <div>
              <div className="gradcam-img-wrap">
                <img
                  src={`data:image/png;base64,${result.gradcam_base64}`}
                  alt="Grad-CAM Overlay"
                />
              </div>
              <div className="gradcam-caption">Grad-CAM overlay</div>
            </div>
          </div>
        </div>
      )}

      <div className="threshold-note">
        threshold: {result.threshold} (Youden's J)
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreview] = useState<string | null>(null);
  const [dragover, setDragover] = useState<boolean>(false);
  const [uiState, setUiState] = useState<UIState>("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const ALLOWED = ["image/jpeg", "image/jpg", "image/png"];

  const handleFile = useCallback((f: File) => {
    setFileError(null);
    setResult(null);
    setUiState("idle");

    if (!ALLOWED.includes(f.type)) {
      setFileError("Only JPEG and PNG files are supported.");
      return;
    }
    if (f.size > 10 * 1e6) {
      setFileError("File exceeds 10 MB limit.");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragover(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setUiState("loading");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      setUiState("success");
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : "Analysis failed. Please try again.";
      setResult({
        error: errorMsg,
        prediction: "NORMAL", // dummy required field fallbacks for typescript match
        confidence: 0,
        pneumonia_prob: 0,
        normal_prob: 0,
        threshold: 0
      });
      setUiState("error");
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setUiState("idle");
    setFileError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app">

        {/* Header */}
        <header className="header">
          <div className="header-title">
            <span>Butyag</span>
          </div>
          <div className="header-sub">Chest X-Ray Screening</div>
          <div className="header-badge">EfficientNet-B3 · ONNX</div>
        </header>

        {/* Main layout */}
        <div className="layout">

          {/* Left: Upload */}
          <div className="upload-panel">
            <div className="panel-label">X-Ray Image</div>

            <div
              className={`dropzone ${dragover ? "dragover" : ""} ${previewUrl ? "has-image" : ""}`}
              onClick={() => !previewUrl && inputRef.current?.click()}
              onDragOver={(e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragover(true); }}
              onDragLeave={() => setDragover(false)}
              onDrop={onDrop}
            >
              {/* Scan-line signature element */}
              {!previewUrl && <div className="scanline" />}

              {previewUrl ? (
                <img src={previewUrl} alt="Uploaded X-Ray" className="preview-img" />
              ) : (
                <>
                  <UploadIcon />
                  <div className="dropzone-text">
                    Drop X-ray here or{" "}
                    <strong onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>browse</strong>
                  </div>
                  <div className="dropzone-hint">JPEG · PNG · max 10 MB</div>
                </>
              )}
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              style={{ display: "none" }}
              onChange={(e: ChangeEvent<HTMLInputElement>) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />

            {fileError && <div className="error-msg">⚠ {fileError}</div>}

            {previewUrl && (
              <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
                <button
                  className="analyze-btn"
                  style={{ flex: 1 }}
                  onClick={analyze}
                  disabled={uiState === "loading"}
                >
                  {uiState === "loading" ? "Analyzing..." : "Analyze X-Ray"}
                </button>
                <button
                  className="analyze-btn"
                  style={{
                    flex: "0 0 auto", width: 44,
                    background: COLORS.navyLight,
                    color: COLORS.slate,
                  }}
                  onClick={reset}
                  title="Clear"
                >
                  ✕
                </button>
              </div>
            )}

            {!previewUrl && (
              <button
                className="analyze-btn"
                style={{ marginTop: 16 }}
                disabled
              >
                Analyze X-Ray
              </button>
            )}
          </div>

          {/* Right: Results */}
          <ResultPanel
            state={uiState}
            result={result}
            previewUrl={previewUrl}
          />
        </div>

        {/* Disclaimer */}
        <div className="disclaimer">
          <strong>Clinical disclaimer:</strong> Butyag is an AI screening aid designed
          to assist, not replace, clinical judgment. All results must be reviewed and
          confirmed by a licensed physician before any diagnosis or treatment decision
          is made. This tool has not been cleared by the FDA or equivalent regulatory body.
        </div>

      </div>
    </>
  );
}