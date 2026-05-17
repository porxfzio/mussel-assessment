import { useState } from "react";
import "./style.css";
import ConveyorMode from "./ConveyorMode";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function getSessionId() {
  return crypto.randomUUID();
}

function ScanningSlot({ file, label }) {
  return (
    <div className="scanning-wrapper">
      <img src={URL.createObjectURL(file)} alt={label} />
      <div className="scan-overlay" />
      <div className="scan-line" />
      <div className="scan-corner tl" />
      <div className="scan-corner tr" />
      <div className="scan-corner bl" />
      <div className="scan-corner br" />
      <div className="scan-label">ANALYSING</div>
    </div>
  );
}

function UploadIcon() {
  return (
    <div className="slot-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3v13.5M12 3l4.5 4.5M12 3L7.5 7.5" />
        <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5" />
      </svg>
    </div>
  );
}

function cap(value) {
  return Math.min(Math.max(value ?? 0, 0), 100);
}

// ── Label color — red for worst grade, default dark otherwise ──
function labelColor(isWorst) {
  return isWorst ? "#c0392b" : "#2c2c2a";
}

export default function App() {
  const [sessionId] = useState(getSessionId);
  const [shellA, setShellA] = useState(null);
  const [shellB, setShellB] = useState(null);
  const [meat, setMeat] = useState(null);
  const [initialResult, setInitialResult] = useState(null);
  const [finalResult, setFinalResult] = useState(null);
  const [loadingInitial, setLoadingInitial] = useState(false);
  const [loadingFinal, setLoadingFinal] = useState(false);
  const [conveyorMode, setConveyorMode] = useState(false);

  async function getInitialGrade() {
    const form = new FormData();
    form.append("shell_a", shellA);
    form.append("shell_b", shellB);
    setLoadingInitial(true);
    try {
      const res = await fetch(`${API_BASE}/predict/initial/${sessionId}`, { method: "POST", body: form });
      const data = await res.json();
      setInitialResult(data);
    } catch (err) { alert("Error getting initial grade"); }
    setLoadingInitial(false);
  }

 async function getFinalGrade() {
  const form = new FormData();
  form.append("meat", meat);
  setLoadingFinal(true);
  try {
    const res = await fetch(`${API_BASE}/predict/final/${sessionId}`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.text();
      alert("Error getting final grade: " + err);
      setLoadingFinal(false);
      return;
    }
    const data = await res.json();
    setFinalResult(data);
  } catch (err) {
    alert("Error getting final grade");
  }
  setLoadingFinal(false);
}

  const getGradeClass = (g) => g === "A" ? "gc-a" : g === "B" ? "gc-b" : "gc-c";
  const getBadgeClass = (g) => g === "A" ? "gb-a" : g === "B" ? "gb-b" : "gb-c";

  // Add this right at the top of your return, before <div className="app">
  if (conveyorMode) {
    return <ConveyorMode onBack={() => setConveyorMode(false)} />;
  }

  return (
    <div className="app">

      {/* ── HEADER ── */}
      <div className="header">
        <p className="eyebrow">A Computer Vision-Based Quality Assessment</p>
        <h1 className="page-title">Green Mussel Quality Assessment</h1>
        <p className="page-sub">
          Upload shell exterior photos for an initial grade based on biofouling,
          then upload the opened mussel photo for the final grade.
        </p>

        {/* Conveyor belt mode button — top right */}
        <button
          onClick={() => setConveyorMode(true)}
          style={{
            position: "absolute",
            top: 0, right: 0,
            display: "flex", alignItems: "center", gap: 6,
            background: "#0F6E56", color: "#E1F5EE",
            border: "none", borderRadius: 8,
            padding: "9px 16px", fontSize: 12,
            fontWeight: 600, cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          🎥 Conveyor Belt Mode
        </button>
      </div>



      {/* ── STEPPER ── */}
      <div className="stepper">
        <div className={`step ${!initialResult ? "active" : "done"}`}>
          <div className="step-num">1</div>
          <div className="step-label">Shell photos</div>
        </div>
        <div className={`step-line ${initialResult ? "done" : ""}`}></div>
        <div className={`step ${initialResult && !finalResult ? "active" : finalResult ? "done" : ""}`}>
          <div className="step-num">2</div>
          <div className="step-label">Initial grade</div>
        </div>
        <div className={`step-line ${finalResult ? "done" : ""}`}></div>
        <div className={`step ${initialResult && !finalResult ? "active" : finalResult ? "done" : ""}`}>
          <div className="step-num">3</div>
          <div className="step-label">Meat photo</div>
        </div>
        <div className="step-line"></div>
        <div className={`step ${finalResult ? "active" : ""}`}>
          <div className="step-num">4</div>
          <div className="step-label">Final grade</div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          STAGE 1 — Shell photos
      ══════════════════════════════════════════ */}
      <div className="card">
        <div className="card-header">
          <span className="tag tag-ext">Stage 1 — External</span>
          <div>
            <p className="card-title">Upload shell exterior photos</p>
            <p className="card-sub">Both sides of the closed mussel shell</p>
          </div>
        </div>

        <div className="slots-grid two-col">
          <div className="slot" onClick={() => !loadingInitial && document.getElementById("fileA").click()}>
            {loadingInitial && shellA ? (
              <div style={{ width: "100%", aspectRatio: "1/1", overflow: "hidden", borderRadius: "8px" }}>
                <ScanningSlot file={shellA} label="Side A" />
              </div>
            ) : shellA ? (
              <div className="slot-filled"><img src={URL.createObjectURL(shellA)} alt="Side A" /></div>
            ) : (
              <div className="slot-empty"><UploadIcon /><p className="slot-name">Side A</p><p className="slot-hint">First side of shell</p></div>
            )}
            <input id="fileA" type="file" accept="image/*" onChange={(e) => setShellA(e.target.files[0])} />
          </div>

          <div className="slot" onClick={() => !loadingInitial && document.getElementById("fileB").click()}>
            {loadingInitial && shellB ? (
              <div style={{ width: "100%", aspectRatio: "1/1", overflow: "hidden", borderRadius: "8px" }}>
                <ScanningSlot file={shellB} label="Side B" />
              </div>
            ) : shellB ? (
              <div className="slot-filled"><img src={URL.createObjectURL(shellB)} alt="Side B" /></div>
            ) : (
              <div className="slot-empty"><UploadIcon /><p className="slot-name">Side B</p><p className="slot-hint">Another side of shell</p></div>
            )}
            <input id="fileB" type="file" accept="image/*" onChange={(e) => setShellB(e.target.files[0])} />
          </div>
        </div>

        <button className="btn-primary" disabled={!shellA || !shellB || loadingInitial} onClick={getInitialGrade}>
          {loadingInitial ? "Analysing..." : "Get initial grade"}
        </button>

        {initialResult && (() => {
          const bioCoverage = cap(initialResult.features.bio_coverage_pct);
          return (
            <div className="result-area" style={{ display: "block" }}>
              <div className="result-top">
                <div className="grade-row">
                  <div className={`grade-circle ${getGradeClass(initialResult.grade)}`}>{initialResult.grade}</div>
                  <div className="grade-info">
                    <span className={`grade-badge ${getBadgeClass(initialResult.grade)}`}>Grade {initialResult.grade}</span>
                    <p className="grade-title">Initial Grade</p>
                    <p className="grade-desc">Biofouling coverage: {bioCoverage.toFixed(1)}%</p>
                    {initialResult.broken_shell && (
                      <p style={{ fontSize: "11px", color: "#c0392b", fontWeight: 600, marginTop: "6px" }}>⚠ Broken shell detected</p>
                    )}
                  </div>
                </div>
                <div className="prob-card">
                  <p className="prob-title">Grade probabilities</p>
                  {["A", "B", "C"].map((g) => (
                    <div className="prob-row" key={g}>
                      <span className="prob-lbl">Grade {g}</span>
                      <div className="prob-track">
                        <div className={`prob-fill ${g === "A" ? "green" : g === "B" ? "amber" : "red"}`}
                          style={{ width: `${cap(initialResult.probabilities[g] * 100).toFixed(1)}%` }} />
                      </div>
                      <span className="prob-pct">{cap(initialResult.probabilities[g] * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "1rem" }}>
                {["a", "b"].map((side) => (
                  <div key={side}>
                    <p style={{ fontSize: "12px", fontWeight: 500, marginBottom: "6px", color: "#5f5e5a" }}>Side {side.toUpperCase()}</p>
                    <div style={{ borderRadius: "8px", overflow: "hidden", border: "0.5px solid #d3d1c7" }}>
                      <img src={`data:image/png;base64,${initialResult[`overlay_${side}`]}`} style={{ width: "100%", display: "block" }} alt={`Side ${side.toUpperCase()} overlay`} />
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginTop: "10px", padding: "10px 12px", background: "#f7f6f3", borderRadius: "8px", border: "0.5px solid #d3d1c7" }}>
                {[
                  { color: "rgba(100,200,100,0.6)", label: "Shell" },
                  { color: "rgba(220,80,80,0.7)",   label: "Attached Biofouling" },
                  { color: "rgba(220,180,60,0.6)",  label: "Residual Biofouling" },
                  { color: "rgba(200,120,200,0.6)", label: "Meat" },
                ].map(({ color, label }) => (
                  <div key={label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <div style={{ width: "12px", height: "12px", borderRadius: "3px", background: color, flexShrink: 0 }} />
                    <span style={{ fontSize: "11px", color: "#5f5e5a" }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}
      </div>

      {/* ══════════════════════════════════════════
          STAGE 2 — Meat photo
      ══════════════════════════════════════════ */}
      <div className={`card ${!initialResult ? "locked" : ""}`}>
        <div className="card-header">
          <span className="tag tag-int">Stage 2 — Internal</span>
          <div>
            <p className="card-title">Upload opened mussel</p>
            <p className="card-sub">Dorsal view with meat fully exposed</p>
          </div>
        </div>

        <div className={`slot wide ${meat ? "has-file" : ""}`} onClick={() => !loadingFinal && document.getElementById("fileMeat").click()}>
          {loadingFinal && meat ? (
            <div style={{ width: "50%", aspectRatio: "1/1", overflow: "hidden", borderRadius: "8px" }}>
              <ScanningSlot file={meat} label="Meat" />
            </div>
          ) : meat ? (
            <div className="slot-filled">
              <img src={URL.createObjectURL(meat)} alt="meat preview" />
              <button className="remove-btn" onClick={(e) => { e.stopPropagation(); setMeat(null); }}>✕</button>
            </div>
          ) : (
            <div className="slot-empty">
              <UploadIcon />
              <p className="slot-name">Opened mussel — meat exposed</p>
              <p className="slot-hint">Clear top-down photo of shucked mussel</p>
            </div>
          )}
          <input id="fileMeat" type="file" accept="image/*" onChange={(e) => setMeat(e.target.files[0])} />
        </div>

        <button className="btn-primary btn-green" disabled={!meat || !initialResult || loadingFinal} onClick={getFinalGrade}>
          {loadingFinal ? "Analysing..." : "Get final grade"}
        </button>

        {finalResult && (() => {
          const meatRatio    = cap(finalResult.features.meat_ratio_display);
          const colorDev     = cap(finalResult.features.flesh_color_dev);
          const bioCoverage  = cap(finalResult.features.bio_coverage_pct);
          const weightApprox = finalResult.features.meat_yield_weight_approx || "—";
          const yieldLabel   = finalResult.features.meat_yield_label          || "Unknown";
          const colorLabel   = finalResult.features.flesh_color_label         || "Unknown";
          const colorDevLbl  = finalResult.features.flesh_color_dev_label     || "Unknown";

          // ── Bio label and color ───────────────────────────────────────────
          const bioLbl      = bioCoverage <= 5 ? "No Biofouling"
                            : bioCoverage <= 25 ? "Light Biofouling"
                            : bioCoverage <= 60 ? "Moderate Biofouling"
                            : "Heavy Biofouling";
          const isHeavyBio  = bioCoverage > 60;
          const isHighColor = colorDevLbl === "High Color Deviation";
          const isLowYield  = yieldLabel === "Low meat yield";

          return (
            <div className="result-area" style={{ display: "block" }}>

              {/* Overlay image */}
              <div style={{ display: "flex", justifyContent: "center", marginBottom: "16px" }}>
                <div style={{ width: "500px", aspectRatio: "1 / 1", borderRadius: "10px", overflow: "hidden", border: "0.5px solid #d3d1c7" }}>
                  <img src={`data:image/png;base64,${finalResult.overlay_meat}`} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} alt="meat overlay" />
                </div>
              </div>

              {/* 3-metric strip */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                border: "0.5px solid #d3d1c7",
                borderRadius: "8px",
                overflow: "hidden",
                marginBottom: "16px",
              }}>

                {/* Shell-to-Meat Ratio */}
                <div style={{ padding: "14px 16px", textAlign: "center", borderRight: "0.5px solid #d3d1c7", background: "#faf9f7" }}>
                  <p style={{ fontSize: "13px", color: "#555", marginBottom: "6px", fontWeight: 700 }}>
                    Shell-to-Meat Ratio
                  </p>
                  <p style={{ fontSize: "9px", color: "#aaa", marginBottom: "6px", fontStyle: "italic" }}>
                    pixel-based estimate
                  </p>
                  <p style={{ fontSize: "20px", fontWeight: 700, color: "#2c2c2a", fontVariantNumeric: "tabular-nums" }}>
                    {meatRatio.toFixed(1)}%
                  </p>
                  <p style={{ fontSize: "13px", color: "#999", marginTop: "4px", fontStyle: "italic" }}>
                    est. real yield ≈ {weightApprox}
                  </p>
                  <p style={{ fontSize: "15px", fontWeight: 700, color: labelColor(isLowYield), marginTop: "5px" }}>
                    {yieldLabel}
                  </p>
                </div>

                {/* Color Deviation */}
                <div style={{ padding: "14px 16px", textAlign: "center", borderRight: "0.5px solid #d3d1c7", background: "#faf9f7" }}>
                  <p style={{ fontSize: "13px", color: "#555", marginBottom: "6px", fontWeight: 700 }}>Color Deviation</p>
                  <p style={{ fontSize: "9px", color: "#aaa", marginBottom: "6px", fontStyle: "italic" }}>
                    CIE DeltaE76
                  </p>
                  <p style={{ fontSize: "20px", fontWeight: 700 }}>
                    ΔE {finalResult.color_dev_raw}
                  </p>
                  <p style={{ fontSize: "13px", color: "#888", marginTop: "2px" }}>
                    {finalResult.color_dev_pct}% (approx.)
                  </p>  
                  <p style={{ fontSize: "15px", fontWeight: 700, color: labelColor(isHighColor), marginTop: "5px" }}>
                    {colorDevLbl}
                  </p>
                </div>

                {/* Biofouling Coverage */}
                <div style={{ padding: "14px 16px", textAlign: "center", background: "#faf9f7" }}>
                  <p style={{ fontSize: "13px", color: "#555", marginBottom: "6px", fontWeight: 700 }}>Biofouling Coverage</p>
                  <p style={{ fontSize: "9px", color: "#aaa", marginBottom: "6px", fontStyle: "italic" }}>
                    ‎ 
                  </p>
                  <p style={{ fontSize: "20px", fontWeight: 700, color: "#2c2c2a", fontVariantNumeric: "tabular-nums" }}>
                    {bioCoverage.toFixed(1)}%
                  </p>
                  <p style={{ fontSize: "15px", fontWeight: 700, color: labelColor(isHeavyBio), marginTop: "5px" }}>
                    {bioLbl}
                  </p>
                </div>

              </div>

              {/* Weighted score breakdown */}
              <div style={{ border: "0.5px solid #d3d1c7", borderRadius: "8px", padding: "16px 20px", marginBottom: "16px", background: "#faf9f7" }}>
                <p style={{ fontSize: "12px", fontWeight: 700, color: "#2c2c2a", marginBottom: "14px" }}>How the final grade was determined</p>
                <div style={{ borderTop: "0.5px solid #d3d1c7", paddingTop: "12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", gap: "16px" }}>
                    {[
                      { label: "Meat Ratio", weight: 0.40, value: meatRatio,   higher: true  },
                      { label: "Color",      weight: 0.30, value: colorDev,    higher: false },
                      { label: "Biofouling", weight: 0.30, value: bioCoverage, higher: false },
                    ].map(({ label, weight, value, higher }) => {
                      const score = higher ? Math.min(value / 100, 1.0) : Math.max(0, 1.0 - value / 100);
                      const contribution = (score * weight * 100).toFixed(1);
                      return (
                        <div key={label} style={{ textAlign: "center" }}>
                          <p style={{ fontSize: "10px", color: "#888", marginBottom: "2px" }}>{label}</p>
                          <p style={{ fontSize: "13px", fontWeight: 700, color: "#2c2c2a" }}>{contribution}%</p>
                          <p style={{ fontSize: "9px", color: "#aaa" }}>of {(weight * 100).toFixed(0)}%</p>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <p style={{ fontSize: "10px", color: "#888", marginBottom: "2px" }}>Overall score</p>
                    <p style={{ fontSize: "18px", fontWeight: 700, color: "#2c2c2a" }}>
                      {((Math.min(meatRatio / 100, 1.0) * 0.40 + Math.max(0, 1.0 - colorDev / 100) * 0.30 + Math.max(0, 1.0 - bioCoverage / 100) * 0.30) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Final grade row */}
              <div style={{ display: "flex", alignItems: "center", gap: "16px", padding: "16px 20px", background: "#faf9f7", borderRadius: "8px", border: "0.5px solid #d3d1c7" }}>
                <div className={`grade-circle large ${getGradeClass(finalResult.grade)}`}>{finalResult.grade}</div>
                <div>
                  <p style={{ fontSize: "11px", color: "#888", fontWeight: 500, marginBottom: "3px" }}>Final Grade</p>
                  <span className={`grade-badge ${getBadgeClass(finalResult.grade)}`}>Grade {finalResult.grade}</span>
                  {finalResult.broken_shell_override && (
                    <p style={{ fontSize: "11px", color: "#8b3a2a", marginTop: "4px" }}>Broken shell detected — forced to Grade C</p>
                  )}
                </div>
                <div style={{ marginLeft: "auto", minWidth: "180px" }}>
                  {["A", "B", "C"].map((g) => (
                    <div className="prob-row" key={g}>
                      <span className="prob-lbl">Grade {g}</span>
                      <div className="prob-track">
                        <div className={`prob-fill ${g === "A" ? "green" : g === "B" ? "amber" : "red"}`}
                          style={{ width: `${cap((finalResult.probabilities[g] ?? 0) * 100).toFixed(1)}%` }} />
                      </div>
                      <span className="prob-pct">{cap((finalResult.probabilities[g] ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Reset button */}
              <div style={{ textAlign: "center", marginTop: "24px", paddingBottom: "16px" }}>
                <button
                  className="btn-primary btn-green"
                  onClick={() => { setShellA(null); setShellB(null); setMeat(null); setInitialResult(null); setFinalResult(null); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                  style={{ padding: "12px 32px", background: "#0F6E56", color: "#E1F5EE", border: "none", borderRadius: "8px", fontSize: "14px", fontWeight: 600, cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.02em" }}
                >
                  Start New Assessment
                </button>
                <p style={{ fontSize: "11px", color: "#888", marginTop: "8px" }}>This will clear all current results and start fresh</p>
              </div>

            </div>
          );
        })()}
      </div>
    </div>
  );
}
