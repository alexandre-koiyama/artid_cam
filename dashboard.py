import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px

LOG_FILE    = Path(__file__).with_name("crossing_log.txt")
OUTPUT_FILE = Path(__file__).with_name("dashboard.html")

LINE_RE = re.compile(
    r"\[Id:(?P<id>\d+), Dir:(?P<direction>IN|OUT), (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
)

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_ORDER   = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]


def load_data(file_path: Path) -> pd.DataFrame:
    rows = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            rows.append({
                "id":        int(m.group("id")),
                "direction": m.group("direction"),
                "timestamp": m.group("timestamp"),
            })
    return pd.DataFrame(rows)


def build_html(df: pd.DataFrame) -> str:
    # embed all records as JSON so JS can filter client-side
    records_json = json.dumps(df.to_dict(orient="records"))
    min_date = df["timestamp"].str[:10].min()
    max_date = df["timestamp"].str[:10].max()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Crossing Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ background: #0e1117; color: #e0e0e0;
              font-family: 'Segoe UI', sans-serif; min-height: 100vh; }}
    header {{ background: #161b22; padding: 20px 32px;
              border-bottom: 1px solid #30363d;
              display: flex; align-items: center; gap: 12px; }}
    header h1 {{ font-size: 1.5rem; color: #58a6ff; }}
    header span {{ font-size: .82rem; color: #8b949e; margin-left: auto; }}

    /* ── filters ── */
    .filters {{ display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap;
                padding: 20px 32px; background: #0d1117;
                border-bottom: 1px solid #30363d; }}
    .fgroup {{ display: flex; flex-direction: column; gap: 4px; }}
    .fgroup label {{ font-size: .75rem; color: #8b949e; text-transform: uppercase; }}
    .filters input, .filters select {{
      background: #21262d; border: 1px solid #30363d; color: #e0e0e0;
      border-radius: 6px; padding: 7px 11px; font-size: .85rem; }}
    .filters input:focus, .filters select:focus {{
      outline: none; border-color: #58a6ff; }}
    .btn {{ background: #238636; border: none; color: #fff;
            border-radius: 6px; padding: 8px 20px; cursor: pointer;
            font-size: .85rem; align-self: flex-end; }}
    .btn:hover {{ background: #2ea043; }}
    .btn.reset {{ background: #30363d; }}
    .btn.reset:hover {{ background: #3d444d; }}

    /* ── kpis ── */
    .kpis {{ display: flex; gap: 14px; padding: 20px 32px; flex-wrap: wrap; }}
    .kpi  {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
             padding: 16px 24px; flex: 1; min-width: 130px; text-align: center; }}
    .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: #58a6ff; }}
    .kpi .lbl {{ font-size: .72rem; color: #8b949e; margin-top: 4px;
                 text-transform: uppercase; letter-spacing: .05em; }}

    /* ── grid ── */
    .grid {{ display: grid; grid-template-columns: 1fr 1fr;
             gap: 18px; padding: 0 32px 28px; }}
    .card {{ background: #161b22; border: 1px solid #30363d;
             border-radius: 10px; padding: 16px; min-height: 360px; }}
    .card.wide {{ grid-column: span 2; min-height: 360px; }}

    /* ── table ── */
    .tbl-wrap {{ max-height: 340px; overflow-y: auto; }}
    table  {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
    thead  {{ position: sticky; top: 0; background: #21262d; z-index: 1; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; text-align: left; }}
    tr:hover td {{ background: #1c2128; }}
    .in  {{ color: #3fb950; font-weight: 600; }}
    .out {{ color: #ff6b6b; font-weight: 600; }}

    @media (max-width: 700px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .card.wide {{ grid-column: span 1; }}
    }}
  </style>
</head>
<body>

<header>
  <h1>📹 Crossing Dashboard</h1>
  <span id="period"></span>
</header>

<!-- filters -->
<div class="filters">
  <div class="fgroup">
    <label>From</label>
    <input type="date" id="f_from" value="{min_date}"/>
  </div>
  <div class="fgroup">
    <label>To</label>
    <input type="date" id="f_to" value="{max_date}"/>
  </div>
  <div class="fgroup">
    <label>Direction</label>
    <select id="f_dir">
      <option value="ALL">ALL</option>
      <option value="IN">IN</option>
      <option value="OUT">OUT</option>
    </select>
  </div>
  <div class="fgroup">
    <label>Weekday</label>
    <select id="f_weekday">
      <option value="ALL">ALL</option>
      <option>Monday</option><option>Tuesday</option><option>Wednesday</option>
      <option>Thursday</option><option>Friday</option>
      <option>Saturday</option><option>Sunday</option>
    </select>
  </div>
  <div class="fgroup">
    <label>Month</label>
    <select id="f_month">
      <option value="ALL">ALL</option>
      <option>January</option><option>February</option><option>March</option>
      <option>April</option><option>May</option><option>June</option>
      <option>July</option><option>August</option><option>September</option>
      <option>October</option><option>November</option><option>December</option>
    </select>
  </div>
  <button class="btn" onclick="applyFilters()">Apply</button>
  <button class="btn reset" onclick="resetFilters()">Reset</button>
</div>

<!-- kpis -->
<div class="kpis">
  <div class="kpi"><div class="val" id="k_total">–</div><div class="lbl">Total crossings</div></div>
  <div class="kpi"><div class="val" id="k_ids">–</div><div class="lbl">Unique IDs</div></div>
  <div class="kpi"><div class="val" id="k_in" style="color:#3fb950">–</div><div class="lbl">Total IN</div></div>
  <div class="kpi"><div class="val" id="k_out" style="color:#ff6b6b">–</div><div class="lbl">Total OUT</div></div>
</div>

<!-- charts + table -->
<div class="grid">
  <div class="card" id="chart_weekday"></div>
  <div class="card" id="chart_donut"></div>
  <div class="card wide" id="chart_month"></div>
  <div class="card wide" id="chart_hour"></div>
  <div class="card wide">
    <h3 style="margin-bottom:10px;font-size:.9rem;color:#8b949e">
      Last 50 crossings
    </h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>ID</th><th>Direction</th><th>Timestamp</th></tr></thead>
        <tbody id="tbl_body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
// ── raw data embedded at build time ──────────────────────────────────────────
const RAW = {records_json};

const WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const MONTH_ORDER   = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"];
function darkLayout(extra) {{
  return Object.assign({{
    paper_bgcolor:"#161b22", plot_bgcolor:"#161b22",
    font:{{color:"#e0e0e0"}},
    xaxis:{{gridcolor:"#21262d", linecolor:"#30363d"}},
    yaxis:{{gridcolor:"#21262d", linecolor:"#30363d"}},
    margin:{{t:40,b:30,l:40,r:20}}
  }}, extra || {{}});
}}

// ── helpers ───────────────────────────────────────────────────────────────────
function getWeekday(ts) {{
  return ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
         [new Date(ts).getDay()];
}}
function getMonth(ts) {{
  return ["January","February","March","April","May","June","July","August",
          "September","October","November","December"][new Date(ts).getMonth()];
}}
function getHour(ts) {{ return new Date(ts).getHours(); }}

function avgBy(data, keyFn, order) {{
  // group by date+key → count, then average per key
  const dateKey = {{}};
  data.forEach(r => {{
    const k  = keyFn(r.timestamp);
    const d  = r.timestamp.slice(0,10);
    const id = d + "|" + k;
    dateKey[id] = (dateKey[id] || 0) + 1;
  }});
  const sums = {{}}, counts = {{}};
  Object.entries(dateKey).forEach(([id, v]) => {{
    const k = id.split("|")[1];
    sums[k]   = (sums[k]   || 0) + v;
    counts[k] = (counts[k] || 0) + 1;
  }});
  return order.map(k => ({{ key: k, avg: counts[k] ? +(sums[k]/counts[k]).toFixed(2) : 0 }}));
}}

// ── filter ────────────────────────────────────────────────────────────────────
function filtered() {{
  const from    = document.getElementById("f_from").value;
  const to      = document.getElementById("f_to").value;
  const dir     = document.getElementById("f_dir").value;
  const weekday = document.getElementById("f_weekday").value;
  const month   = document.getElementById("f_month").value;

  return RAW.filter(r => {{
    const d = r.timestamp.slice(0,10);
    if (from   && d < from) return false;
    if (to     && d > to)   return false;
    if (dir    !== "ALL" && r.direction !== dir)          return false;
    if (weekday !== "ALL" && getWeekday(r.timestamp) !== weekday) return false;
    if (month   !== "ALL" && getMonth(r.timestamp)   !== month)   return false;
    return true;
  }});
}}

// ── render ────────────────────────────────────────────────────────────────────
function render(data) {{
  // kpis
  const ids = new Set(data.map(r => r.id));
  document.getElementById("k_total").textContent = data.length;
  document.getElementById("k_ids").textContent   = ids.size;
  document.getElementById("k_in").textContent    = data.filter(r => r.direction==="IN").length;
  document.getElementById("k_out").textContent   = data.filter(r => r.direction==="OUT").length;

  const minD = data.length ? data[0].timestamp.slice(0,10)  : "–";
  const maxD = data.length ? data[data.length-1].timestamp.slice(0,10) : "–";
  document.getElementById("period").textContent = minD + " → " + maxD;

  // weekday bar
  const wk = avgBy(data, ts => getWeekday(ts), WEEKDAY_ORDER);
  Plotly.react("chart_weekday", [{{
    type:"bar", x: wk.map(d=>d.key), y: wk.map(d=>d.avg),
    marker:{{ color: wk.map(d=>d.avg), colorscale:"Blues" }}
  }}], darkLayout({{title:{{text:"Avg Visits by Weekday"}}}}));

  // month bar
  const mo = avgBy(data, ts => getMonth(ts), MONTH_ORDER);
  Plotly.react("chart_month", [{{
    type:"bar", x: mo.map(d=>d.key), y: mo.map(d=>d.avg),
    marker:{{ color: mo.map(d=>d.avg), colorscale:"Teal" }}
  }}], darkLayout({{title:{{text:"Avg Visits by Month"}}}}));

  // hour line
  const hrMap = {{}};
  const hrCnt = {{}};
  data.forEach(r => {{
    const d = r.timestamp.slice(0,10);
    const h = getHour(r.timestamp);
    const k = d+"|"+h;
    hrMap[k] = (hrMap[k]||0)+1;
  }});
  Object.entries(hrMap).forEach(([k,v]) => {{
    const h = parseInt(k.split("|")[1]);
    hrCnt[h] = {{ s:(hrCnt[h]?.s||0)+v, c:(hrCnt[h]?.c||0)+1 }};
  }});
  const hours = [...Array(24).keys()];
  const hrAvg = hours.map(h => hrCnt[h] ? +(hrCnt[h].s/hrCnt[h].c).toFixed(2) : 0);
  Plotly.react("chart_hour", [{{
    type:"scatter", mode:"lines+markers",
    x: hours, y: hrAvg,
    line:{{color:"#00d4ff"}}, marker:{{color:"#00d4ff"}}
  }}], darkLayout({{
    title:{{text:"Avg Visits by Hour"}},
    xaxis:{{dtick:1, gridcolor:"#21262d", linecolor:"#30363d"}}
  }}));

  // donut
  const inC  = data.filter(r=>r.direction==="IN").length;
  const outC = data.filter(r=>r.direction==="OUT").length;
  Plotly.react("chart_donut", [{{
    type:"pie", hole:0.55,
    labels:["IN","OUT"], values:[inC, outC],
    marker:{{colors:["#00d4ff","#ff6b6b"]}}
  }}], darkLayout({{title:{{text:"IN vs OUT"}}}}));

  // table – last 50
  const tbody = document.getElementById("tbl_body");
  const rows  = data.slice(-50).reverse();
  tbody.innerHTML = rows.map(r =>
    `<tr>
      <td>${{r.id}}</td>
      <td class="${{r.direction.toLowerCase()}}">${{r.direction}}</td>
      <td>${{r.timestamp}}</td>
    </tr>`
  ).join("");
}}

function applyFilters() {{ render(filtered()); }}
function resetFilters() {{
  document.getElementById("f_from").value    = "{min_date}";
  document.getElementById("f_to").value      = "{max_date}";
  document.getElementById("f_dir").value     = "ALL";
  document.getElementById("f_weekday").value = "ALL";
  document.getElementById("f_month").value   = "ALL";
  render(RAW);
}}

// initial render
render(RAW);
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    if not LOG_FILE.exists():
        print(f"❌ File not found: {LOG_FILE}")
        raise SystemExit(1)

    print(f"📂 Reading {LOG_FILE} ...")
    df = load_data(LOG_FILE)

    if df.empty:
        print("❌ No valid records found.")
        raise SystemExit(1)

    print(f"✅ {len(df)} records loaded.")
    html = build_html(df)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard saved → {OUTPUT_FILE}")

    import webbrowser
    webbrowser.open(OUTPUT_FILE.as_uri())