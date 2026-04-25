"""Tiny no-websocket dashboard served directly by FastAPI."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Return a lightweight polling dashboard that avoids Streamlit websockets."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AutoResearch Chess Lab Live</title>
  <style>
    body { margin: 0; background: #0f1117; color: #f5f5f5; font: 14px system-ui, sans-serif; }
    header { padding: 22px 28px; border-bottom: 1px solid #2a2f3a; display: flex; gap: 20px; align-items: end; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 30px; }
    main { padding: 20px 28px; display: grid; gap: 18px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }
    .metric, section { background: #171a22; border: 1px solid #2a2f3a; border-radius: 8px; padding: 14px; }
    .label { color: #aab1c3; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 26px; margin-top: 6px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #2a2f3a; padding: 8px; text-align: left; vertical-align: top; }
    th { color: #aab1c3; font-weight: 600; }
    code, pre { white-space: pre-wrap; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr); gap: 18px; }
    .pill { display: inline-block; border: 1px solid #3b4354; border-radius: 999px; padding: 2px 8px; color: #cbd5f1; }
    .muted { color: #aab1c3; }
    @media (max-width: 900px) { .grid, .metrics { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>AutoResearch Chess Lab</h1>
    <span class="muted">FastAPI live dashboard, no Streamlit websocket</span>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><div class="label">Experiment</div><div id="exp" class="value">...</div></div>
      <div class="metric"><div class="label">Status</div><div id="status" class="value">...</div></div>
      <div class="metric"><div class="label">Current Arm</div><div id="arm" class="value">...</div></div>
      <div class="metric"><div class="label">Tokens</div><div id="tokens" class="value">0</div></div>
    </div>
    <div class="grid">
      <section>
        <h2>Research Trace</h2>
        <div id="trace"></div>
      </section>
      <section>
        <h2>Leaderboard</h2>
        <div id="leaderboard"></div>
      </section>
    </div>
    <div class="grid">
      <section>
        <h2>Artifacts</h2>
        <div id="artifacts"></div>
      </section>
      <section>
        <h2>Budget</h2>
        <div id="budget"></div>
      </section>
    </div>
  </main>
  <script>
    const short = (v, n=180) => {
      const s = typeof v === "string" ? v : JSON.stringify(v);
      return s && s.length > n ? s.slice(0, n) + " ..." : (s || "");
    };
    const table = (rows, cols) => {
      if (!rows || !rows.length) return "<p class='muted'>No rows yet.</p>";
      return `<table><thead><tr>${cols.map(c => `<th>${c[0]}</th>`).join("")}</tr></thead><tbody>` +
        rows.map(r => `<tr>${cols.map(c => `<td>${short(r[c[1]])}</td>`).join("")}</tr>`).join("") +
        "</tbody></table>";
    };
    async function get(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + " " + r.status);
      return await r.json();
    }
    async function refresh() {
      try {
        const exps = await get("/experiments");
        if (!exps.length) return;
        exps.sort((a,b) => b.created_at.localeCompare(a.created_at));
        const exp = exps[0];
        document.getElementById("exp").textContent = exp.id;
        document.getElementById("status").textContent = exp.status;
        document.getElementById("arm").textContent = exp.current_arm || "";
        const [budget, trace, leaderboard, artifacts] = await Promise.all([
          get(`/experiments/${exp.id}/budget`),
          get(`/experiments/${exp.id}/research-trace?limit=80`),
          get(`/experiments/${exp.id}/leaderboard`),
          get(`/experiments/${exp.id}/artifact-summaries`)
        ]);
        document.getElementById("tokens").textContent = budget.reduce((n,r) => n + (r.actual_tokens || r.estimated_tokens || 0), 0).toLocaleString();
        document.getElementById("trace").innerHTML = trace.slice(-40).reverse().map(e =>
          `<p><b>${e.created_at}</b> <span class="pill">${e.event_type}</span><br>${e.message}<br><code>${short(e.payload, 320)}</code></p>`
        ).join("") || "<p class='muted'>No trace yet.</p>";
        document.getElementById("leaderboard").innerHTML = table(leaderboard, [
          ["Arm", "arm"], ["Name", "name"], ["Arch", "architecture"], ["Status", "status"], ["Score", "score"], ["Tokens", "tokens_spent"]
        ]);
        document.getElementById("artifacts").innerHTML = table(artifacts.slice(-30).reverse(), [
          ["Type", "artifact_type"], ["Title", "title"], ["Arm", "arm"], ["Preview", "content_preview"]
        ]);
        document.getElementById("budget").innerHTML = table(budget.slice(-20).reverse(), [
          ["Arm", "experiment_arm"], ["Pod", "pod_id"], ["Role", "agent_role"], ["Candidate", "candidate_id"], ["Tokens", "estimated_tokens"]
        ]);
      } catch (err) {
        console.error(err);
      }
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""
