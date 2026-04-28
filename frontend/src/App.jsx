import { useState } from "react";
import "./style.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function getSessionId() {
  let id = localStorage.getItem("mussel_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("mussel_session_id", id);
  }
  return id;
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

  // ── API CALLS ──
  async function getInitialGrade() {
    const form = new FormData();
    form.append("shell_a", shellA);
    form.append("shell_b", shellB);

    setLoadingInitial(true);

    try {
      const res = await fetch(`${API_BASE}/predict/initial/${sessionId}`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setInitialResult(data);
    } catch (err) {
      alert("Error getting initial grade");
    }

    setLoadingInitial(false);
  }

  async function getFinalGrade() {
    const form = new FormData();
    form.append("meat", meat);

    setLoadingFinal(true);

    try {
      const res = await fetch(`${API_BASE}/predict/final/${sessionId}`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setFinalResult(data);
    } catch (err) {
      alert("Error getting final grade");
    }

    setLoadingFinal(false);
  }

  // ── HELPERS ──
  const handleFile = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;

    if (type === "a") setShellA(file);
    if (type === "b") setShellB(file);
    if (type === "meat") setMeat(file);
  };

  const getGradeClass = (grade) => {
    if (grade === "A") return "gc-a";
    if (grade === "B") return "gc-b";
    return "gc-c";
  };

  const getBadgeClass = (grade) => {
    if (grade === "A") return "gb-a";
    if (grade === "B") return "gb-b";
    return "gb-c";
  };

  return (
    <div className="app">
      {/* HEADER */}
      <div className="header">
        <p className="eyebrow">Cavite State University · BSCS Thesis</p>
        <h1 className="page-title">Green Mussel Quality Assessment</h1>
        <p className="page-sub">
          Upload shell exterior photos for an initial grade based on biofouling,
          then upload the opened mussel photo for the final grade.
        </p>
      </div>

      {/* STEPPER */}
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

        <div className={`step ${finalResult ? "active" : ""}`}>
          <div className="step-num">3</div>
          <div className="step-label">Meat photo</div>
        </div>

        <div className="step-line"></div>

        <div className="step">
          <div className="step-num">4</div>
          <div className="step-label">Final grade</div>
        </div>
      </div>


      {/* STAGE 1 */}
      <div className="card">
        <div className="card-header">
          <span className="tag tag-ext">Stage 1 — External</span>
          <div>
            <p className="card-title">Upload shell exterior photos</p>
            <p className="card-sub">
              Both sides of the closed mussel shell
            </p>
          </div>
        </div>

        <div className="slots-grid two-col">
          <div className="slot" onClick={() => document.getElementById("fileA").click()}>
            
            {!shellA ? (
              <div className="slot-empty">
                <div className="slot-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M12 3v13.5M12 3l4.5 4.5M12 3L7.5 7.5" />
                    <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5" />
                  </svg>
                </div>
                <p className="slot-name">Side A</p>
                <p className="slot-hint">First side of shell</p>
              </div>
            ) : (
              <div className="slot-filled">
                <img src={URL.createObjectURL(shellA)} alt="preview" />
              </div>
            )}

            <input
              id="fileA"
              type="file"
              accept="image/*"
              onChange={(e) => setShellA(e.target.files[0])}
            />
          </div>
          <div className="slot" onClick={() => document.getElementById("fileB").click()}>
            
            {!shellB ? (
              <div className="slot-empty">
                <div className="slot-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M12 3v13.5M12 3l4.5 4.5M12 3L7.5 7.5" />
                    <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5" />
                  </svg>
                </div>
                <p className="slot-name">Side B</p>
                <p className="slot-hint">Another side of shell</p>
              </div>
            ) : (
              <div className="slot-filled">
                <img src={URL.createObjectURL(shellB)} alt="preview" />
              </div>
            )}

            <input
              id="fileB"
              type="file"
              accept="image/*"
              onChange={(e) => setShellB(e.target.files[0])}
            />
          </div>


        </div>

        <button
          className="btn-primary"
          disabled={!shellA || !shellB}
          onClick={getInitialGrade}
        >
          {loadingInitial ? "Analysing..." : "Get initial grade"}
        </button>

        {/* RESULT */}
        {initialResult && (
          <div className="result-area" style={{ display: "block" }}>

            {/* ROW 1: Grade + Probability bars */}
            <div className="result-top">
              <div className="grade-row">
                <div className={`grade-circle ${getGradeClass(initialResult.grade)}`}>
                  {initialResult.grade}
                </div>
                <div className="grade-info">
                  <span className={`grade-badge ${getBadgeClass(initialResult.grade)}`}>
                    Grade {initialResult.grade}
                  </span>
                  <p className="grade-title">Initial Grade</p>
                  <p className="grade-desc">
                    Biofouling coverage: {(initialResult.features.bio_coverage_pct).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="prob-card">
                <p className="prob-title">Grade probabilities</p>
                {["A", "B", "C"].map((g) => (
                  <div className="prob-row" key={g}>
                    <span className="prob-lbl">Grade {g}</span>
                    <div className="prob-track">
                      <div
                        className={`prob-fill ${g === "A" ? "green" : g === "B" ? "amber" : "red"}`}
                        style={{ width: `${(initialResult.probabilities[g] * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <span className="prob-pct">
                      {(initialResult.probabilities[g] * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* ROW 2: Side A | Side B */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "1rem" }}>
              {["a", "b"].map((side) => (
                <div key={side}>
                  <p style={{ fontSize: "12px", fontWeight: 500, marginBottom: "6px", color: "#5f5e5a" }}>
                    Side {side.toUpperCase()}
                  </p>
                  <div style={{ borderRadius: "8px", overflow: "hidden", border: "0.5px solid #d3d1c7" }}>
                    <img
                      src={`data:image/png;base64,${initialResult[`overlay_${side}`]}`}
                      style={{ width: "100%", display: "block" }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
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
        )}
      </div>

      {/* STAGE 2 */}
      <div className={`card ${!initialResult ? "locked" : ""}`}>
        <div className="card-header">
          <span className="tag tag-int">Stage 2 — Internal</span>
          <div>
            <p className="card-title">Upload opened mussel</p>
          </div>
        </div>

        <div
          className={`slot wide ${meat ? "has-file" : ""}`}
          onClick={() => document.getElementById("fileMeat").click()}
        >
          {/* EMPTY STATE (icon + text) */}
          {!meat && (
            <div className="slot-empty">
              <div className="slot-icon">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5" />
                  <path d="M12 3l4.5 4.5M12 3L7.5 7.5M12 3v13.5" />
                </svg>
              </div>

              <p className="slot-name">Opened mussel — meat exposed</p>
              <p className="slot-hint">
                Clear top-down photo of shucked mussel
              </p>
            </div>
          )}

          {/* FILLED STATE (preview) */}
          {meat && (
            <div className="slot-filled">
              <img src={URL.createObjectURL(meat)} alt="meat preview" />
              <button
                className="remove-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setMeat(null);
                }}
              >
                ✕
              </button>
            </div>
          )}

          {/* HIDDEN INPUT */}
          <input
            id="fileMeat"
            type="file"
            accept="image/*"
            onChange={(e) => setMeat(e.target.files[0])}
          />
        </div>


        <button
          className="btn-primary btn-green"
          disabled={!meat || !initialResult}
          onClick={getFinalGrade}
        >
          {loadingFinal ? "Analysing..." : "Get final grade"}
        </button>

          {finalResult && (
            <div className="result-area" style={{ display: "block" }}>
              {/* Overlay image */}
              <div style={{
                display: "flex",
                justifyContent: "center",
                marginBottom: "16px"
              }}>
                <div style={{
                  width: "500px",          // ← control the size here
                  aspectRatio: "1 / 1",
                  borderRadius: "10px",
                  overflow: "hidden",
                  border: "0.5px solid #d3d1c7",
                }}>
                  <img
                    src={`data:image/png;base64,${finalResult.overlay_meat}`}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      display: "block",
                    }}
                    alt="meat segmentation"
                  />
                </div>
              </div>



            {/* 3-metric row */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              border: "0.5px solid #d3d1c7",
              borderRadius: "8px",
              overflow: "hidden",
              marginBottom: "16px"
            }}>
              {[
                {
                  label: "Shell-to-meat ratio",
                  value: `${(finalResult.features.meat_shell_ratio ?? 0).toFixed(1)}%`,
                },
                {
                  label: "Color deviation",
                  value: `${(finalResult.features.flesh_color_dev ?? 0).toFixed(2)}`,
                },
                {
                  label: "Biofouling coverage",
                  value: `${(finalResult.features.bio_coverage_pct ?? 0).toFixed(1)}%`,
                },
              ].map(({ label, value }, i, arr) => (
                <div
                  key={label}
                  style={{
                    padding: "14px 16px",
                    textAlign: "center",
                    borderRight: i < arr.length - 1 ? "0.5px solid #d3d1c7" : "none",
                    background: "#faf9f7",
                  }}
                >
                  <p style={{ fontSize: "11px", color: "#888", marginBottom: "6px", fontWeight: 500 }}>
                    {label}
                  </p>
                  <p style={{ fontSize: "20px", fontWeight: 700, color: "#2c2c2a", fontVariantNumeric: "tabular-nums" }}>
                    {value}
                  </p>
                </div>
              ))}
            </div>

            {/* Final grade row */}
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: "16px",
              padding: "16px 20px",
              background: "#faf9f7",
              borderRadius: "8px",
              border: "0.5px solid #d3d1c7",
            }}>
              <div className={`grade-circle large ${getGradeClass(finalResult.grade)}`}>
                {finalResult.grade}
              </div>
              <div>
                <p style={{ fontSize: "11px", color: "#888", fontWeight: 500, marginBottom: "3px" }}>
                  Final Grade
                </p>
                <span className={`grade-badge ${getBadgeClass(finalResult.grade)}`}>
                  Grade {finalResult.grade}
                </span>
                {finalResult.broken_shell_override && (
                  <p style={{ fontSize: "11px", color: "#8b3a2a", marginTop: "4px" }}>
                    Broken shell detected — forced to Grade C
                  </p>
                )}
              </div>

              {/* Probability bars on the right */}
              <div style={{ marginLeft: "auto", minWidth: "180px" }}>
                {["A", "B", "C"].map((g) => (
                  <div className="prob-row" key={g}>
                    <span className="prob-lbl">Grade {g}</span>
                    <div className="prob-track">
                      <div
                        className={`prob-fill ${g === "A" ? "green" : g === "B" ? "amber" : "red"}`}
                        style={{ width: `${((finalResult.probabilities[g] ?? 0) * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <span className="prob-pct">
                      {((finalResult.probabilities[g] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
