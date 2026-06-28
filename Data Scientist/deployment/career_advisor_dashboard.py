"""
Dashboard: Indonesia Data & AI Talent Market Intelligence
Jalankan dengan: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ID Data & AI Talent Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }

  /* Header */
  .header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 24px;
    margin-bottom: 24px;
  }
  .header-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.05em;
  }
  .header-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #e6edf3;
  }
  .header-filter {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8b949e;
  }

  /* Section label */
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* Scorecard */
  .scorecard {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
  }
  .scorecard-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 10px;
  }
  .scorecard-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
  }
  .scorecard-value.green  { color: #3fb950; }
  .scorecard-value.orange { color: #d29922; }

  /* Divider */
  .divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 28px 0;
  }

  /* Status badge */
  .badge-ready      { background:#1a3a2a; color:#3fb950; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
  .badge-gap        { background:#3a2a10; color:#d29922; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }

  /* Table */
  .styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .styled-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #30363d;
  }
  .styled-table td {
    padding: 12px 14px;
    border-bottom: 1px solid #161b22;
    color: #c9d1d9;
    vertical-align: middle;
  }
  .styled-table tr:hover td { background: #161b22; }
  .styled-table .mono { font-family: 'JetBrains Mono', monospace; }

  /* Gauge container */
  .gauge-wrap {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
  }

  /* Hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 24px; padding-bottom: 24px; }
</style>
""", unsafe_allow_html=True)

# ─── MOCK DATA ────────────────────────────────────────────────────────────────
# Ganti bagian ini nanti dengan data real dari pipeline RAG-mu

ROLE_OPTIONS = ["All Roles", "Data Scientist", "Data Analyst", "AI Engineer", "Data Engineer"]

talent_data = pd.DataFrame([
    {"ID": "T-001", "Role": "Data Analyst",   "Skill": "Python",     "Salary": "Rp 8.500.000",  "Score": 85, "Status": "READY"},
    {"ID": "T-002", "Role": "Data Scientist",  "Skill": "Statistics", "Salary": "Rp 19.000.000", "Score": 40, "Status": "GAP DETECT"},
    {"ID": "T-003", "Role": "AI Engineer",     "Skill": "MLOps",      "Salary": "Rp 25.000.000", "Score": 90, "Status": "READY"},
    {"ID": "T-004", "Role": "Data Engineer",   "Skill": "Spark",      "Salary": "Rp 15.000.000", "Score": 70, "Status": "READY"},
    {"ID": "T-005", "Role": "Data Scientist",  "Skill": "Deep Learning","Salary":"Rp 22.000.000", "Score": 55, "Status": "GAP DETECT"},
])

skills_data = {"Python": 65, "MLOps": 50, "Cloud (AWS)": 35, "Big Data": 25, "Statistics": 15}

role_scores = {"Data Scientist": 72, "Data Analyst": 85, "AI Engineer": 90, "Data Engineer": 68}

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-logo">◈ TALENIQ</div>
  <div class="header-title">Dashboard Indonesia Data &amp; AI Talent Market Intelligence</div>
  <div class="header-filter">FILTER: ROLE ▾</div>
</div>
""", unsafe_allow_html=True)

# ─── FILTER ───────────────────────────────────────────────────────────────────
selected_role = st.selectbox("", ROLE_OPTIONS, label_visibility="collapsed")

# Filter data
if selected_role != "All Roles":
    filtered = talent_data[talent_data["Role"] == selected_role]
else:
    filtered = talent_data

# ─── MAIN METRICS ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Main Metrics</div>', unsafe_allow_html=True)

avg_score = int(filtered["Score"].mean()) if not filtered.empty else 0
total_cv   = len(filtered)
# Simulasi avg salary (ambil angka dari string)
avg_salary = filtered["Salary"].str.replace(r"[^0-9]", "", regex=True).astype(int).mean()
avg_salary_fmt = f"Rp {avg_salary:,.0f}".replace(",", ".")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="scorecard">
      <div class="scorecard-label">Total CV Analyzed</div>
      <div class="scorecard-value">{total_cv:,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    color = "green" if avg_score >= 70 else "orange"
    st.markdown(f"""
    <div class="scorecard">
      <div class="scorecard-label">Score CV Readiness</div>
      <div class="scorecard-value {color}">{avg_score}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="scorecard">
      <div class="scorecard-label">Avg Salary Benchmark</div>
      <div class="scorecard-value orange">{avg_salary_fmt}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── MARKET DRILL-DOWN ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Market Drill-Down</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

# ── GAUGE: Role Match ──────────────────────────────────────────────────────────
with left_col:
    gauge_role  = selected_role if selected_role != "All Roles" else "Data Scientist"
    gauge_value = role_scores.get(gauge_role, 72)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={"suffix": "%", "font": {"size": 36, "color": "#58a6ff", "family": "JetBrains Mono"}},
        title={"text": f"Role Match — {gauge_role}", "font": {"size": 13, "color": "#8b949e"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e", "size": 10}},
            "bar": {"color": "#58a6ff", "thickness": 0.28},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  50], "color": "#1a1f26"},
                {"range": [50, 75], "color": "#1a2a1a"},
                {"range": [75,100], "color": "#1a3a2a"},
            ],
            "threshold": {"line": {"color": "#3fb950", "width": 3}, "value": 70},
        }
    ))
    fig_gauge.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#161b22",
        font_color="#e6edf3",
    )
    st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # Rekomendasi teks singkat di bawah gauge
    if gauge_value >= 80:
        rec = "✅ Kandidat sangat siap untuk role ini."
    elif gauge_value >= 60:
        rec = "⚠️ Kandidat cukup siap, ada beberapa gap minor."
    else:
        rec = "❌ Perlu peningkatan signifikan sebelum apply."
    st.markdown(f"<p style='font-size:12px;color:#8b949e;text-align:center;margin-top:0'>{rec}</p>",
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── BAR CHART: Top 5 Skills ────────────────────────────────────────────────────
with right_col:
    skills  = list(skills_data.keys())
    values  = list(skills_data.values())
    colors  = ["#58a6ff" if v == max(values) else "#30363d" for v in values]

    fig_bar = go.Figure(go.Bar(
        x=values,
        y=skills,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont={"family": "JetBrains Mono", "size": 12, "color": "#c9d1d9"},
    ))
    fig_bar.update_layout(
        title={"text": "Top 5 Skills in Demand", "font": {"size": 13, "color": "#8b949e"}},
        height=260,
        margin=dict(t=40, b=10, l=20, r=50),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(
            autorange="reversed",
            tickfont={"color": "#c9d1d9", "size": 12, "family": "Inter"},
            gridcolor="#21262d",
        ),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── CV AUDIT TABLE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">CV Section Audit Breakdown</div>', unsafe_allow_html=True)

def score_bar(val):
    filled = int(val / 10)
    bar = "█" * filled + "░" * (10 - filled)
    color = "#3fb950" if val >= 70 else "#d29922"
    return f'<span style="font-family:JetBrains Mono;font-size:11px;color:{color}">{bar}</span> {val}%'

def status_badge(s):
    if s == "READY":
        return '<span class="badge-ready">READY</span>'
    return '<span class="badge-gap">GAP DETECT</span>'

rows_html = ""
for _, row in filtered.iterrows():
    rows_html += f"""
    <tr>
      <td class="mono" style="color:#58a6ff">{row['ID']}</td>
      <td>{row['Role']}</td>
      <td>{row['Skill']}</td>
      <td class="mono">{row['Salary']}</td>
      <td>{score_bar(row['Score'])}</td>
      <td>{status_badge(row['Status'])}</td>
    </tr>
    """

st.markdown(f"""
<table class="styled-table">
  <thead>
    <tr>
      <th>ID Talent</th>
      <th>Role Category</th>
      <th>Skill</th>
      <th>Expected Salary</th>
      <th>Matching Score</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Data diperbarui otomatis dari pipeline RAG · Taleniq v0.1")
