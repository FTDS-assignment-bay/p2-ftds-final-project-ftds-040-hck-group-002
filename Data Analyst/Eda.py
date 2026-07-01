import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")

# MEMUAT DATA
try:
    df = pd.read_csv('jobs_final.csv', sep=';')
except FileNotFoundError:
    st.error("File 'jobs_final.csv' tidak ditemukan di folder ini.")
    st.stop()

# DATA PREVIEW
st.subheader("📋 Ringkasan Dataset")
st.dataframe(df.head(10)) 

# Pre-processing Data Gaji
df['avg_salary'] = (df['min_salary'].fillna(0) + df['max_salary'].fillna(0)) / 2

# VISUALISASI
st.header("Visualisasi Data")

# ANALISIS ROLE
st.subheader("💼 Analisis Kategori Role")

def map_to_4_roles(title):
    t = str(title).lower()
    if 'scientist' in t or 'science' in t:
        return 'Data Scientist'
    elif 'analyst' in t or 'analytic' in t or 'intelligence' in t:
        return 'Data Analyst'
    elif 'engineer' in t:
        return 'Data Engineer'
    else:
        return 'AI Engineer'

df['role_category'] = df['job_title'].apply(map_to_4_roles)
role_counts = df['role_category'].value_counts().reset_index()
role_counts.columns = ['Role Kategori', 'Jumlah']

c1, c2 = st.columns(2)
with c1:
    fig_pie = px.pie(role_counts, values='Jumlah', names='Role Kategori', hole=0.4,
                     title='Proporsi Lowongan',
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig_pie, use_container_width=True)
with c2:
    fig_bar = px.bar(role_counts, x='Role Kategori', y='Jumlah', title='Jumlah Lowongan per Role',
                     color='Role Kategori', text_auto=True,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ANALISIS GAJI
st.subheader("💰 Analisis Gaji")

# Hitung statistik
stats = df[df['avg_salary'] > 0].groupby('role_category')['avg_salary'].agg(['mean', 'min', 'max']).reset_index()

c3, c4 = st.columns(2)

with c3:
    # Gaji Tertinggi (Bar Chart)
    gaji_max = df.groupby('role_category')['max_salary'].max().reset_index()
    fig_max = px.bar(gaji_max, x='role_category', y='max_salary', 
                     title='Gaji Tertinggi per Role', 
                     text_auto=True,
                     color='role_category', 
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig_max.update_layout(template='plotly_white', showlegend=False)
    st.plotly_chart(fig_max, use_container_width=True)

with c4:
    # Gaji Rata-rata (Floating Range + Mean Diamond)
    fig = go.Figure()
    # Garis Rentang (Min-Max)
    fig.add_trace(go.Bar(x=stats['role_category'], y=stats['max'] - stats['min'], base=stats['min'],
                         marker_color='#4C78A8', opacity=0.4, name='Rentang Gaji'))
    # Titik Rata-rata (Mean)
    fig.add_trace(go.Scatter(x=stats['role_category'], y=stats['mean'], mode='markers',
                             marker=dict(size=14, symbol='diamond', color='#FF9F00'), name='Rata-rata'))
    
    fig.update_layout(title='Gaji Rata-rata & Rentang', template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ANALISIS SKILL
st.subheader("🛠️ Top 10 Skill Paling Dicari")
skills_to_track = ['excel', 'sql', 'python', 'dashboard', 'cloud', 
                   'power bi', 'git', 'tableau', 'etl', 'statistics', 
                   'machine learning', 'data warehouse', 'mlops']
skills_counter = Counter()
for desc in df['job_description'].dropna():
    for skill in skills_to_track:
        if skill in desc.lower(): skills_counter[skill] += 1

top_skills = pd.DataFrame(skills_counter.most_common(10), columns=['Skill', 'Total'])
fig_skills = px.bar(top_skills, x='Total', y='Skill', orientation='h', 
                    title='Skill Teratas', color='Total',
                    color_continuous_scale='Plasma')

fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_skills, use_container_width=True)