import { useState, useRef, useEffect } from "react";

const API_BASE = import.meta.env?.VITE_API_URL || "http://localhost:8000";

const STAGES = ["side_a", "side_b", "meat"];
const STAGE_LABELS = { side_a: "Side A", side_b: "Side B", meat: "Meat" };
const STAGE_TAGS = { side_a: "Stage 1 — External", side_b: "Stage 1 — External", meat: "Stage 2 — Internal" };

const STATUS_COLORS = {
  waiting:    "#b4b2a9",
  processing: "#EF9F27",
  saved:      "#0F6E56",
  error:      "#E24B4A",
};

const STATUS_DOT = ({ status }) => {
  const color = STATUS_COLORS[status] || STATUS_COLORS.waiting;
  return (
    <span style={{
      display: "inline-block",
      width: 8, height: 8,
      borderRadius: "50%",
      background: color,
      marginRight: 6,
      flexShrink: 0,
      boxShadow: status === "processing" ? `0 0 6px ${color}` : "none",
      animation: status === "processing" ? "pulse 1.2s ease-in-out infinite" : "none",
    }} />
  );
};

const GradeCircle = ({ grade }) => {
  const colors = {
    A: { bg: "#E1F5EE", color: "#085041" },
    B: { bg: "#FAEEDA", color: "#633806" },
    C: { bg: "#FCEBEB", color: "#791F1F" },
  };
  const c = colors[grade] || { bg: "#f0f0f0", color: "#888" };
  return (
    <div style={{
      width: 32, height: 32,
      borderRadius: "50%",
      background: c.bg,
      color: c.color,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontWeight: 700, fontSize: 14,
      flexShrink: 0,
    }}>
      {grade}
    </div>
  );
};

export default function ConveyorMode() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [currentStage, setCurrentStage] = useState("side_a");
  const [snapping, setSnapping]         = useState(false);
  const [processing, setProcessing]     = useState(false);
  const [snapPreview, setSnapPreview]   = useState(null);

  // mussels[0..2] = { id, side_a, side_b, meat, initial_grade, final_grade, status_a, status_b, status_meat }
  const [mussels, setMussels] = useState([
    { id: "Mussel_01", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
    { id: "Mussel_02", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
    { id: "Mussel_03", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
  ]);

  const [log, setLog] = useState([]);

  const addLog = (msg) => setLog(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 29)]);

  // ── Snap photo from video ──────────────────────────────────────────────────
  const snapAndCrop = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;

    const w = video.videoWidth  || video.clientWidth;
    const h = video.videoHeight || video.clientHeight;
    canvas.width  = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    // Full frame preview
    setSnapPreview(canvas.toDataURL("image/jpeg", 0.92));

    // Crop into 3 equal horizontal sections
    const cropW = Math.floor(w / 3);
    const crops = [0, 1, 2].map(i => {
      const offscreen = document.createElement("canvas");
      offscreen.width  = cropW;
      offscreen.height = h;
      offscreen.getContext("2d").drawImage(canvas, i * cropW, 0, cropW, h, 0, 0, cropW, h);
      return offscreen.toDataURL("image/jpeg", 0.92);
    });

    return crops;
  };

  const dataURLtoBlob = (dataURL) => {
    const [header, data] = dataURL.split(",");
    const mime = header.match(/:(.*?);/)[1];
    const bytes = atob(data);
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return new Blob([arr], { type: mime });
  };

  // ── Process snap ──────────────────────────────────────────────────────────
  const handleSnap = async () => {
    if (snapping || processing) return;
    setSnapping(true);

    const crops = snapAndCrop();
    if (!crops) { setSnapping(false); return; }

    addLog(`Snapped frame — stage: ${STAGE_LABELS[currentStage]}`);
    setSnapping(false);
    setProcessing(true);

    // Update status to processing for all 3 mussels
    setMussels(prev => prev.map((m, i) => ({
      ...m,
      [`status_${currentStage === "side_a" ? "a" : currentStage === "side_b" ? "b" : "meat"}`]: "processing",
    })));

    if (currentStage === "side_a") {
      await processSideA(crops);
    } else if (currentStage === "side_b") {
      await processSideB(crops);
    } else {
      await processMeat(crops);
    }

    setProcessing(false);
  };

  // ── Side A ────────────────────────────────────────────────────────────────
  const processSideA = async (crops) => {
    const updated = [...mussels];
    for (let i = 0; i < 3; i++) {
      const sessionId = crypto.randomUUID();
      updated[i] = { ...updated[i], side_a: crops[i], session_id: sessionId };
      addLog(`${updated[i].id} Side A → saved (session: ${sessionId.slice(0,8)}...)`);
      updated[i].status_a = "saved";
      setMussels([...updated]);
    }
  };

  // ── Side B ────────────────────────────────────────────────────────────────
  const processSideB = async (crops) => {
    const updated = [...mussels];
    for (let i = 0; i < 3; i++) {
      updated[i] = { ...updated[i], side_b: crops[i] };
      addLog(`${updated[i].id} Side B → processing initial grade...`);
      setMussels([...updated]);

      try {
        const form = new FormData();
        const blobA = dataURLtoBlob(updated[i].side_a);
        const blobB = dataURLtoBlob(crops[i]);
        form.append("shell_a", blobA, "shell_a.jpg");
        form.append("shell_b", blobB, "shell_b.jpg");

        const res  = await fetch(`${API_BASE}/predict/initial/${updated[i].session_id}`, { method: "POST", body: form });
        const data = await res.json();

        updated[i] = {
          ...updated[i],
          initial_grade: data.grade,
          status_b: "saved",
          initial_data: data,
        };
        addLog(`${updated[i].id} Initial Grade → ${data.grade} (bio: ${data.features?.bio_coverage_pct?.toFixed(1)}%)`);
      } catch (err) {
        updated[i].status_b = "error";
        addLog(`${updated[i].id} ERROR: ${err.message}`);
      }
      setMussels([...updated]);
    }
  };

  // ── Meat ──────────────────────────────────────────────────────────────────
  const processMeat = async (crops) => {
    const updated = [...mussels];
    for (let i = 0; i < 3; i++) {
      updated[i] = { ...updated[i], meat: crops[i] };
      addLog(`${updated[i].id} Meat → processing final grade...`);
      setMussels([...updated]);

      try {
        const form = new FormData();
        const blob = dataURLtoBlob(crops[i]);
        form.append("meat", blob, "meat.jpg");

        const res  = await fetch(`${API_BASE}/predict/final/${updated[i].session_id}`, { method: "POST", body: form });
        const data = await res.json();

        updated[i] = {
          ...updated[i],
          final_grade: data.grade,
          status_meat: "saved",
          final_data: data,
        };
        addLog(`${updated[i].id} Final Grade → ${data.grade}`);
      } catch (err) {
        updated[i].status_meat = "error";
        addLog(`${updated[i].id} ERROR: ${err.message}`);
      }
      setMussels([...updated]);
    }
  };

  const handleStageChange = (stage) => {
    setCurrentStage(stage);
    setSnapPreview(null);
    addLog(`Stage changed to: ${STAGE_LABELS[stage]}`);
  };

  const handleReset = () => {
    setMussels([
      { id: "Mussel_01", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
      { id: "Mussel_02", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
      { id: "Mussel_03", side_a: null, side_b: null, meat: null, initial_grade: null, final_grade: null, status_a: "waiting", status_b: "waiting", status_meat: "waiting", session_id: null },
    ]);
    setCurrentStage("side_a");
    setSnapPreview(null);
    setLog([]);
    addLog("System reset.");
  };

  const statusKey = (stage) => stage === "side_a" ? "status_a" : stage === "side_b" ? "status_b" : "status_meat";

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: "#f0ede8", minHeight: "100vh", padding: "1.5rem" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
        .mussel-row { animation: fadeIn 0.3s ease; }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: "1.25rem" }}>
        <p style={{ fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "#888780", marginBottom: 4 }}>
          Cavite State University · BSCS Thesis
        </p>
        <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, fontWeight: 400, color: "#1a1a18", margin: 0 }}>
          Green Mussel Quality Assessment
        </h1>
        <p style={{ fontSize: 12, color: "#5f5e5a", marginTop: 4 }}>
          Conveyor Belt Mode — Simulation
        </p>
      </div>

      {/* Stage Selector */}
      <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 12, padding: "0.875rem 1.25rem", marginBottom: "1rem", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "#888780", fontWeight: 500 }}>CURRENT STAGE:</span>
        {STAGES.map(s => (
          <button key={s} onClick={() => handleStageChange(s)} style={{
            padding: "6px 16px", borderRadius: 99, border: "0.5px solid",
            borderColor: currentStage === s ? "#0F6E56" : "#d3d1c7",
            background: currentStage === s ? "#0F6E56" : "#fff",
            color: currentStage === s ? "#E1F5EE" : "#5f5e5a",
            fontSize: 12, fontWeight: 500, cursor: "pointer",
            fontFamily: "'DM Sans', sans-serif",
            transition: "all 0.2s",
          }}>
            {STAGE_LABELS[s]}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button onClick={handleReset} style={{
          padding: "6px 16px", borderRadius: 8, border: "0.5px solid #d3d1c7",
          background: "#fff", color: "#5f5e5a", fontSize: 12, cursor: "pointer",
          fontFamily: "'DM Sans', sans-serif",
        }}>
          Reset
        </button>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>

        {/* LEFT — Video + Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

          {/* Video Panel */}
          <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px", borderBottom: "0.5px solid #f0ede8", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99, background: "#E1F5EE", color: "#085041" }}>
                {STAGE_TAGS[currentStage]}
              </span>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#1a1a18" }}>
                {STAGE_LABELS[currentStage]} — Live Feed
              </span>
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 11, color: "#888780" }}>3 mussels in frame</span>
            </div>

            {/* Video */}
            <div style={{ position: "relative", background: "#1a1a18" }}>
              <video
                ref={videoRef}
                src="/videos/sample.mp4"
                autoPlay loop muted playsInline
                style={{ width: "100%", display: "block", maxHeight: 280, objectFit: "cover" }}
              />
              {/* Zone markers */}
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, display: "flex", pointerEvents: "none" }}>
                {["L", "C", "R"].map((label, i) => (
                  <div key={i} style={{
                    flex: 1,
                    borderRight: i < 2 ? "1px dashed rgba(255,255,255,0.3)" : "none",
                    display: "flex", alignItems: "flex-start", justifyContent: "center",
                    paddingTop: 8,
                  }}>
                    <span style={{ fontSize: 10, color: "rgba(255,255,255,0.5)", background: "rgba(0,0,0,0.3)", padding: "2px 6px", borderRadius: 4 }}>
                      Mussel 0{i + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Snap button */}
            <div style={{ padding: "12px 16px" }}>
              <button
                onClick={handleSnap}
                disabled={snapping || processing}
                style={{
                  width: "100%", padding: "11px", borderRadius: 8, border: "none",
                  background: snapping || processing ? "#d3d1c7" : "#0F6E56",
                  color: snapping || processing ? "#888780" : "#E1F5EE",
                  fontSize: 13, fontWeight: 500, cursor: snapping || processing ? "not-allowed" : "pointer",
                  fontFamily: "'DM Sans', sans-serif", transition: "background 0.15s",
                }}
              >
                {processing ? "Processing..." : snapping ? "Snapping..." : `📸 Snap — ${STAGE_LABELS[currentStage]}`}
              </button>
            </div>
          </div>

          {/* Snap Preview */}
          {snapPreview && (
            <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
              <div style={{ padding: "10px 16px", borderBottom: "0.5px solid #f0ede8" }}>
                <span style={{ fontSize: 12, fontWeight: 500, color: "#1a1a18" }}>Last Snap Preview</span>
              </div>
              <img src={snapPreview} alt="snap" style={{ width: "100%", display: "block", maxHeight: 160, objectFit: "cover" }} />
            </div>
          )}

          {/* Activity Log */}
          <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ padding: "10px 16px", borderBottom: "0.5px solid #f0ede8" }}>
              <span style={{ fontSize: 12, fontWeight: 500, color: "#1a1a18" }}>Activity Log</span>
            </div>
            <div style={{ padding: "10px 16px", maxHeight: 160, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
              {log.length === 0 && (
                <p style={{ fontSize: 11, color: "#b4b2a9", fontStyle: "italic" }}>No activity yet...</p>
              )}
              {log.map((entry, i) => (
                <p key={i} style={{ fontSize: 11, color: i === 0 ? "#1a1a18" : "#888780", margin: 0, fontFamily: "monospace" }}>
                  {entry}
                </p>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT — Queue + Results */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

          {/* Stage 1 — Side A */}
          <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px", borderBottom: "0.5px solid #f0ede8", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99, background: "#E1F5EE", color: "#085041" }}>Stage 1</span>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#1a1a18" }}>Side A</span>
            </div>
            <div style={{ padding: "10px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
              {mussels.map((m, i) => (
                <div key={i} className="mussel-row" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, background: "#fafaf8", border: "0.5px solid #f0ede8" }}>
                  <STATUS_DOT status={m.status_a} />
                  <span style={{ fontSize: 12, fontWeight: 500, color: "#1a1a18", flex: 1 }}>{m.id}</span>
                  <span style={{ fontSize: 11, color: STATUS_COLORS[m.status_a] || "#888780", fontWeight: 500 }}>
                    {m.status_a === "waiting" ? "Waiting..." : m.status_a === "processing" ? "Processing..." : m.status_a === "saved" ? "✓ Saved" : "Error"}
                  </span>
                  {m.side_a && (
                    <img src={m.side_a} alt="" style={{ width: 36, height: 28, objectFit: "cover", borderRadius: 4, border: "0.5px solid #d3d1c7" }} />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Stage 1 — Side B + Initial Grade */}
          <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px", borderBottom: "0.5px solid #f0ede8", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99, background: "#E1F5EE", color: "#085041" }}>Stage 1</span>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#1a1a18" }}>Side B + Initial Grade</span>
            </div>
            <div style={{ padding: "10px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
              {mussels.map((m, i) => (
                <div key={i} className="mussel-row" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, background: "#fafaf8", border: "0.5px solid #f0ede8" }}>
                  <STATUS_DOT status={m.status_b} />
                  <span style={{ fontSize: 12, fontWeight: 500, color: "#1a1a18", flex: 1 }}>{m.id}</span>
                  <span style={{ fontSize: 11, color: STATUS_COLORS[m.status_b] || "#888780", fontWeight: 500 }}>
                    {m.status_b === "waiting" ? "Waiting..." : m.status_b === "processing" ? "Processing..." : m.status_b === "saved" ? "✓ Saved" : "Error"}
                  </span>
                  {m.initial_grade && <GradeCircle grade={m.initial_grade} />}
                  {m.side_b && (
                    <img src={m.side_b} alt="" style={{ width: 36, height: 28, objectFit: "cover", borderRadius: 4, border: "0.5px solid #d3d1c7" }} />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Stage 2 — Meat + Final Grade */}
          <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px", borderBottom: "0.5px solid #f0ede8", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 99, background: "#FAEEDA", color: "#633806" }}>Stage 2</span>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#1a1a18" }}>Meat + Final Grade</span>
            </div>
            <div style={{ padding: "10px 16px", display: "flex", flexDirection: "column", gap: 8 }}>
              {mussels.map((m, i) => (
                <div key={i} className="mussel-row" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 8, background: "#fafaf8", border: "0.5px solid #f0ede8" }}>
                  <STATUS_DOT status={m.status_meat} />
                  <span style={{ fontSize: 12, fontWeight: 500, color: "#1a1a18", flex: 1 }}>{m.id}</span>
                  <span style={{ fontSize: 11, color: STATUS_COLORS[m.status_meat] || "#888780", fontWeight: 500 }}>
                    {m.status_meat === "waiting" ? "Waiting..." : m.status_meat === "processing" ? "Processing..." : m.status_meat === "saved" ? "✓ Saved" : "Error"}
                  </span>
                  {m.final_grade && <GradeCircle grade={m.final_grade} />}
                  {m.meat && (
                    <img src={m.meat} alt="" style={{ width: 36, height: 28, objectFit: "cover", borderRadius: 4, border: "0.5px solid #d3d1c7" }} />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Summary */}
          {mussels.some(m => m.final_grade) && (
            <div style={{ background: "#fff", border: "0.5px solid #d3d1c7", borderRadius: 14, overflow: "hidden" }}>
              <div style={{ padding: "12px 16px", borderBottom: "0.5px solid #f0ede8" }}>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#1a1a18" }}>Batch Summary</span>
              </div>
              <div style={{ padding: "12px 16px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {mussels.map((m, i) => (
                  <div key={i} style={{ textAlign: "center", padding: "10px 8px", borderRadius: 8, background: "#fafaf8", border: "0.5px solid #f0ede8" }}>
                    <p style={{ fontSize: 11, color: "#888780", marginBottom: 6 }}>{m.id}</p>
                    <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 4 }}>
                      {m.initial_grade && (
                        <div>
                          <p style={{ fontSize: 9, color: "#b4b2a9", marginBottom: 2 }}>Initial</p>
                          <GradeCircle grade={m.initial_grade} />
                        </div>
                      )}
                      {m.final_grade && (
                        <div>
                          <p style={{ fontSize: 9, color: "#b4b2a9", marginBottom: 2 }}>Final</p>
                          <GradeCircle grade={m.final_grade} />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for cropping */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
