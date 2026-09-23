"""
FastAPI application for PRISM-Phish.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.inference import PhishingInferenceEngine
from src.api.schemas import EmailAnalysisRequest, EmailAnalysisResponse, HealthCheckResponse

app = FastAPI(
    title="PRISM-Phish API",
    description="Source-Invariant and Perturbation-Consistent Phishing Email Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PhishingInferenceEngine()


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="healthy" if engine.is_ready() else "degraded",
        models_loaded=["logistic_regression", "tfidf_vectorizers"] if engine.is_ready() else [],
        version="1.0.0",
    )


@app.post("/api/v1/analyze", response_model=EmailAnalysisResponse)
def analyze_email(req: EmailAnalysisRequest):
    try:
        return engine.analyze_email(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index_ui():
    """Interactive glassmorphic demo UI for email inspection."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PRISM-Phish — Defensive Email Intelligence</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 48, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --accent-phish: #ff3366;
      --accent-safe: #00e599;
      --accent-blue: #3b82f6;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 2.5rem 1.5rem; background-image: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(255, 51, 102, 0.08) 0px, transparent 50%); }
    .container { max-width: 1050px; margin: 0 auto; }
    header { margin-bottom: 2rem; }
    h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; background: linear-gradient(135deg, #fff 40%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    p.subtitle { color: var(--text-muted); font-size: 1.05rem; margin-top: 0.4rem; }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem; }
    .card { background: var(--card-bg); backdrop-filter: blur(16px); border: 1px solid var(--border); border-radius: 16px; padding: 1.75rem; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
    label { display: block; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 600; }
    input, textarea { width: 100%; background: rgba(13, 19, 33, 0.8); border: 1px solid var(--border); border-radius: 10px; color: var(--text); padding: 0.85rem; font-size: 0.95rem; margin-bottom: 1.2rem; transition: border-color 0.2s; }
    input:focus, textarea:focus { outline: none; border-color: var(--accent-blue); }
    textarea { height: 160px; resize: vertical; }
    button.btn { width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; padding: 0.9rem; border-radius: 10px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
    button.btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4); }
    .badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge.phish { background: rgba(255, 51, 102, 0.15); color: var(--accent-phish); border: 1px solid rgba(255, 51, 102, 0.3); }
    .badge.safe { background: rgba(0, 229, 153, 0.15); color: var(--accent-safe); border: 1px solid rgba(0, 229, 153, 0.3); }
    .metric-row { display: flex; justify-content: space-between; padding: 0.75rem 0; border-bottom: 1px solid var(--border); font-size: 0.95rem; }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
    .signal-tag { background: rgba(255, 255, 255, 0.05); padding: 0.5rem 0.8rem; border-radius: 8px; margin-top: 0.5rem; font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; }
    .signal-tag::before { content: "⚠️"; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>PRISM-Phish Detector</h1>
      <p class="subtitle">Source-Invariant Defensive Intelligence with Hybrid Transformer & Security Heuristics</p>
    </header>

    <div class="grid">
      <div class="card">
        <label for="subj">Email Subject</label>
        <input id="subj" placeholder="e.g., URGENT: Verify your payroll credentials immediately" value="URGENT: Suspicious Login Detected on Your Corporate Account">

        <label for="sender">From Address</label>
        <input id="sender" placeholder="e.g., security@microsoft-verify-help.com" value="it-support@sec-corp-verify.xyz">

        <label for="body">Email Body Content</label>
        <textarea id="body">Dear User,

We noticed an unauthorized login attempt to your workstation from IP 192.168.1.1. Please click the link below to confirm your password credentials within 24 hours:

https://192.168.1.1/login.php?user=urgent

Failure to do so will result in permanent account suspension.

IT Security Helpdesk</textarea>

        <button class="btn" id="analyzeBtn" onclick="runAnalysis()">Inspect Email</button>
      </div>

      <div class="card" id="resultsCard">
        <label>Detection Verdict</label>
        <div style="margin: 0.8rem 0 1.5rem 0;">
          <span class="badge phish" id="verdictBadge">Phishing Detected</span>
        </div>

        <div class="metric-row">
          <span>Phishing Probability</span>
          <span class="metric-val" id="probVal">98.4%</span>
        </div>
        <div class="metric-row">
          <span>Threat Level</span>
          <span class="metric-val" id="riskVal" style="color: #ff3366;">CRITICAL</span>
        </div>
        <div class="metric-row">
          <span>Inference Latency</span>
          <span class="metric-val" id="latencyVal">4.2 ms</span>
        </div>

        <label style="margin-top: 1.5rem;">Detected Threat Signals</label>
        <div id="signalsContainer">
          <div class="signal-tag">Direct IP address found in hyperlink target</div>
          <div class="signal-tag">Link points to high-risk anomalous top-level domain (.xyz)</div>
          <div class="signal-tag">Urgency / coercive language triggers: urgent, account suspension</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function runAnalysis() {
      const btn = document.getElementById("analyzeBtn");
      btn.innerText = "Analyzing...";
      btn.disabled = true;

      const payload = {
        subject: document.getElementById("subj").value,
        sender: document.getElementById("sender").value,
        body: document.getElementById("body").value
      };

      try {
        const res = await fetch("/api/v1/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        const badge = document.getElementById("verdictBadge");
        if (data.prediction === 1) {
          badge.className = "badge phish";
          badge.innerText = "Phishing Detected";
        } else {
          badge.className = "badge safe";
          badge.innerText = "Legitimate Email";
        }

        document.getElementById("probVal").innerText = (data.phishing_probability * 100).toFixed(1) + "%";
        const riskEl = document.getElementById("riskVal");
        riskEl.innerText = data.risk_level;
        riskEl.style.color = (data.risk_level === 'CRITICAL' || data.risk_level === 'HIGH') ? '#ff3366' : '#00e599';
        document.getElementById("latencyVal").innerText = data.inference_time_ms + " ms";

        const sigBox = document.getElementById("signalsContainer");
        sigBox.innerHTML = "";
        if (data.top_signals.length === 0) {
          sigBox.innerHTML = "<div style='color: #94a3b8; font-size: 0.9rem;'>No malicious signals flagged.</div>";
        } else {
          data.top_signals.forEach(s => {
            const el = document.createElement("div");
            el.className = "signal-tag";
            el.innerText = s;
            sigBox.appendChild(el);
          });
        }
      } catch (err) {
        alert("Inference request failed: " + err);
      } finally {
        btn.innerText = "Inspect Email";
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""
