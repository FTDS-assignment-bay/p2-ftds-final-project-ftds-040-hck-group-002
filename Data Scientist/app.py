import streamlit as st
import os
import json
import math
from job_finder import match_cv_to_jobs, extract_cv

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Minimal CSS — only things Streamlit respects
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; }

/* ── Shared html-block styles ── */
.aica-header {
    background: linear-gradient(135deg, #1B3A4B 0%, #2E7D8C 100%);
    padding: 1.1rem 2.2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -4rem 1.5rem -4rem;
}
.aica-header-title { font-size: 1.5rem; font-weight: 700; color: #CAE9F5; letter-spacing: .05em; }
.aica-header-sub   { font-size: .75rem; color: #8BA5B0; font-style: italic; }

.profile-box {
    background: linear-gradient(135deg, #1B3A4B 0%, #2E7D8C 100%);
    border-radius: 12px; padding: 1.1rem 1.5rem; margin-bottom: .25rem;
}
.profile-name { font-size: 1.1rem; font-weight: 700; color: #CAE9F5; }
.profile-role { font-size: .75rem; color: #8BA5B0; margin: 2px 0 6px; }
.profile-summary { font-size: .78rem; color: #D0E8F2; line-height: 1.55; }

.section-label {
    font-size: .65rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #2E7D8C; margin-bottom: .5rem;
}

/* score triplet */
.score-wrap { display: flex; justify-content: space-around; text-align: center; padding: .4rem 0; }
.score-num  { font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 500; color: #1B3A4B; line-height: 1; }
.score-lbl  { font-size: .65rem; color: #8BA5B0; margin-top: 3px; }

/* salary */
.salary-num  { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 500; color: #1B3A4B; }
.salary-sub  { font-size: .7rem; color: #8BA5B0; margin-top: 3px; }

/* skill bar */
.sk-row  { margin-bottom: .6rem; }
.sk-name { font-size: .78rem; font-weight: 500; color: #1B3A4B; display:flex; justify-content:space-between; margin-bottom:3px; }
.sk-val  { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #2E7D8C; }
.sk-bg   { background: #EAF3F8; border-radius: 6px; height: 8px; }
.sk-fill { border-radius: 6px; height: 8px; background: linear-gradient(90deg,#2E7D8C,#CAE9F5); }

/* missing pill */
.pill {
    display:inline-block; background:#FFF3E0; color:#E65100;
    border:1px solid #FFCC80; border-radius:20px;
    padding:3px 12px; font-size:.72rem; font-weight:600;
    margin:3px 4px 3px 0;
}

/* job card */
.job-card {
    border-left: 4px solid #2E7D8C;
    background: #F4FAFD; border-radius: 0 10px 10px 0;
    padding: .75rem 1rem; margin-bottom: .75rem;
}
.job-title-txt { font-size: .9rem; font-weight: 700; color: #1B3A4B; }
.job-company   { font-size: .75rem; color: #2E7D8C; margin: 2px 0; }
.job-meta      { font-size: .7rem; color: #8BA5B0; }
.job-reason    { font-size: .7rem; color: #546E7A; margin-top: 5px; line-height:1.45; }
.match-badge   {
    float:right; background:#1B3A4B; color:#CAE9F5;
    border-radius:8px; padding:2px 9px;
    font-size:.7rem; font-weight:700;
    font-family:'JetBrains Mono',monospace;
}

/* rec card */
.rec-card  { border-radius:10px; padding:.75rem 1rem; margin-bottom:.65rem; }
.rec-high  { background:#FFF3E0; border-left:4px solid #F57C00; }
.rec-med   { background:#E8F5E9; border-left:4px solid #388E3C; }
.rec-low   { background:#EDE7F6; border-left:4px solid #7B1FA2; }
.rec-pri   { font-size:.62rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:3px; }
.pri-high  { color:#F57C00; }
.pri-med   { color:#388E3C; }
.pri-low   { color:#7B1FA2; }
.rec-title { font-size:.85rem; font-weight:700; color:#1B3A4B; }
.rec-desc  { font-size:.72rem; color:#546E7A; margin-top:3px; line-height:1.45; }

/* divider */
.aica-divider { border:none; border-top:1px solid #D6E8F0; margin:1rem 0; }
.aica-footer  { text-align:center; color:#8BA5B0; font-size:.67rem; margin-top:1.5rem; padding-top:.75rem; border-top:1px solid #D6E8F0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_idr(n): return f"Rp {n/1_000_000:.1f} Jt"

def gauge_svg(val, w=180):
    r  = 62
    cx = w // 2
    cy = 78
    half_circ = math.pi * r
    fill = half_circ * (val / 100)
    gap  = half_circ - fill
    col  = "#4CAF82" if val >= 80 else "#2E7D8C" if val >= 60 else "#F57C00"
    h    = cy + 20
    return f"""
<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
        fill="none" stroke="#EAF3F8" stroke-width="13" stroke-linecap="round"/>
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
        fill="none" stroke="{col}" stroke-width="13" stroke-linecap="round"
        stroke-dasharray="{fill:.1f} {gap+10:.1f}"/>
  <text x="{cx}" y="{cy-8}" text-anchor="middle"
        font-family="JetBrains Mono,monospace" font-size="28" font-weight="500" fill="#1B3A4B">{val}</text>
  <text x="{cx}" y="{cy+14}" text-anchor="middle"
        font-family="Inter,sans-serif" font-size="10" fill="#8BA5B0">out of 100</text>
</svg>"""

def skill_bar(name, score):
    return f"""
<div class="sk-row">
  <div class="sk-name"><span>{name}</span><span class="sk-val">{score}</span></div>
  <div class="sk-bg"><div class="sk-fill" style="width:{score}%"></div></div>
</div>"""

def rec_card(rec):
    p   = rec["priority"]
    cls = {"High":"rec-high","Medium":"rec-med","Low":"rec-low"}.get(p,"rec-low")
    pc  = {"High":"pri-high","Medium":"pri-med","Low":"pri-low"}.get(p,"pri-low")
    return f"""
<div class="rec-card {cls}">
  <div class="rec-pri {pc}">{p} · {rec['category']}</div>
  <div class="rec-title">{rec['title']}</div>
  <div class="rec-desc">{rec['description']}</div>
</div>"""

# ── Load data ─────────────────────────────────────────────────────────────────

def load_json(src):
    if hasattr(src, "read"):
        return json.loads(src.read().decode())
    with open(src) as f:
        return json.load(f)


# ── App ───────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="aica-header">
  <div style="font-size:.85rem;font-weight:700;color:#CAE9F5;">🤖 AI.CO</div>
  <div class="aica-header-title">AI Career Advisor</div>
  <div class="aica-header-sub">Your AI Career Companion · Est. 2026</div>
</div>
""", unsafe_allow_html=True)

# ── Row 0 : CV upload + role filter ──────────────────────────────────────────
c0a, c0b = st.columns([1, 1], gap="large")
with c0a:
    st.markdown('<div class="section-label">📄 CV INPUT (JSON)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload JSON hasil analisis CV", type=["pdf"],
                                label_visibility="collapsed")
with c0b:
    st.markdown('<div class="section-label">🎯 FILTER ROLE</div>', unsafe_allow_html=True)
    role_options = ["Data Scientist","Data Engineer","Data Analyst","AI Engineer"]
    role_user = st.selectbox("Pilih role target", role_options, label_visibility="collapsed")

path = ""
if uploaded is not None:
    match_cv_to_jobs(uploaded, role_user)
st.markdown('<hr class="aica-divider">', unsafe_allow_html=True)

# Load data
try:
    data = load_json("output.json")
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

cand  = data["candidate"]
sc    = data["scores"]
sal   = data["salary_prediction"]
jobs  = data["top_jobs"]
skd   = data["skill_analysis"]
recs  = data["recommendations"]

# ── Profile strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="profile-box">
  <div class="profile-name">{cand['name']}</div>
  <div class="profile-role">Target Role: {cand['target_role']}</div>
  <div class="profile-summary">{cand['professional_summary']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="aica-divider">', unsafe_allow_html=True)

# ── Row 1 : CV Score  |  Salary ───────────────────────────────────────────────
c1a, c1b = st.columns([1, 1], gap="large")

with c1a:
    st.markdown('<div class="section-label">📊 CV SCORE</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="score-wrap">
  <div><div class="score-num">{sc['cv_quality']}</div><div class="score-lbl">CV Quality</div></div></div>
</div>""", unsafe_allow_html=True)

with c1b:
    st.markdown('<div class="section-label">💰 SALARY ESTIMATION</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="salary-num">{fmt_idr(sal['average'])}<span style="font-size:1rem;color:#8BA5B0;"> / bulan</span></div>
<div class="salary-sub">Min: {fmt_idr(sal['minimum'])} &nbsp;·&nbsp; Max: {fmt_idr(sal['maximum'])} &nbsp;·&nbsp; {sal['currency']}</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="aica-divider">', unsafe_allow_html=True)

# ── Row 2 : Gauge  |  Matched Skills ──────────────────────────────────────────
c2a, c2b = st.columns([1, 1], gap="large")

with c2a:
    st.markdown('<div class="section-label">🎯 CV RELEVANCE-TO-ROLE SCORE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;justify-content:center;margin-top:.5rem;">{gauge_svg(sc["job_relevance"])}</div>',
        unsafe_allow_html=True)

with c2b:
    st.markdown('<div class="section-label">✅ MATCHED SKILLS</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(skill_bar(s["skill"], s["score"]) for s in skd["matched_skills"]),
        unsafe_allow_html=True)

st.markdown('<hr class="aica-divider">', unsafe_allow_html=True)

# ── Row 3 : Recommendations  |  Missing Skills + Top Jobs ─────────────────────
c3a, c3b = st.columns([1, 1], gap="large")

with c3a:
    st.markdown('<div class="section-label">💡 RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown("".join(rec_card(r) for r in recs), unsafe_allow_html=True)

with c3b:
    # Missing skills
    st.markdown('<div class="section-label">⚠️ MISSING SKILLS</div>', unsafe_allow_html=True)
    pills = "".join(f'<span class="pill">{s}</span>' for s in skd["missing_skills"])
    st.markdown(f'<div style="margin-bottom:.9rem;">{pills}</div>', unsafe_allow_html=True)

    # Top jobs
    st.markdown('<div class="section-label">🏆 TOP JOB MATCHES</div>', unsafe_allow_html=True)
    for job in jobs:
        st.markdown(f"""
<div class="job-card">
  <span class="match-badge">Match {job['match_score']}%</span>
  <div class="job-title-txt">#{job['rank']} {job['job_title']}</div>
  <div class="job-company">{job['company']}</div>
  <div class="job-meta">{job['industry']} · {job['location']} · {fmt_idr(job['estimated_salary'])}/bln</div>
  <div class="job-reason">{job['reason']}</div>
</div>""", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="aica-footer">AI Career Advisor · AI.CO · Your AI Career Companion · Est. 2026</div>',
    unsafe_allow_html=True)
