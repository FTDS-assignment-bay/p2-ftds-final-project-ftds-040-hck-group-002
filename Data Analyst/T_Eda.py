import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re  # Untuk membaca teks gaji yang rumit

st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊Exploratory Data Analysis (EDA)")

# 1. UPLOAD DATA
uploaded_file = st.file_uploader("Upload file dataset di sini (CSV / Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca data
    if uploaded_file.name.endswith('job_details_indonesia.csv'): # Kalo mau coba dataset lain bisa tapi harus sesuaikan u/ kolom-kolomnya
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.success("Data berhasil dimuat!")
    
    # 2. MEMBUAT TABS UNTUK EDA
    tab_overview, tab_viz = st.tabs([
        "📋 Ringkasan Data (Overview)", 
        "📈 Visualisasi Data"
    ])
    
    # TAB 1: OVERVIEW 
    with tab_overview:
        st.header("Ringkasan Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Baris Data", df.shape[0])
        with col2:
            st.metric("Jumlah Kolom/Fitur", df.shape[1])
            
        st.subheader("5 Data Pertama (Head)")
        st.dataframe(df.head())
        
        st.subheader("Cek Missing Values (Data Kosong)")
        missing_data = df.isnull().sum().to_frame(name="Jumlah Kosong")
        st.dataframe(missing_data)

    # TAB 2: VISUALISASI
    with tab_viz:
        st.header("Visualisasi Data")
        st.markdown("---")
        
        # ANALISIS ROLE PEKERJAAN
        if 'job_title' in df.columns:
            st.subheader("💼 Analisis Berdasarkan Kategori Role")
            
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
            role_counts.columns = ['Role Kategori', 'Jumlah Lowongan']

            kolom_pie, kolom_bar_role = st.columns(2)
            
            with kolom_pie:
                fig_pie = px.pie(role_counts, values='Jumlah Lowongan', names='Role Kategori', hole=0.4,
                                 title='Proporsi Lowongan Kerja (Donut Chart)',
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_traces(textinfo='percent+label')
                fig_pie.update_layout(template='plotly_dark', height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with kolom_bar_role:
                fig_bar_role = px.bar(role_counts, x='Role Kategori', y='Jumlah Lowongan', 
                                      title=f'Jumlah Lowongan Kerja Per Role (Total: {df.shape[0]})',
                                      text_auto=True, color='Role Kategori',
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_bar_role.update_layout(template='plotly_dark', showlegend=False, height=380, margin=dict(t=40, b=20, l=10, r=10))
                st.plotly_chart(fig_bar_role, use_container_width=True)
                
            st.markdown("---")
        else:
            st.warning("Kolom 'job_title' tidak ditemukan. Analisis kategori role dilewati.")

        # ANALISIS GAJI 
        if 'salary' in df.columns and 'role_category' in df.columns:
            st.subheader("💰 Analisis Gaji")
            
            # Fungsi pembersih angka teks 
            def bersihkan_gaji(val):
                val = str(val).lower()
                if 'not available' in val or val == 'nan':
                    return None
                    
                # Ekstrak semua format angka ribuan
                nums = re.findall(r'\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b', val)
                # Buang titik agar terbaca sebagai float
                nums = [float(n.replace('.', '').replace(',', '.')) for n in nums]
                if len(nums) > 0:
                # Jika ada 2 angka (Range "Dari - Sampai"), ambil nilai tengah (Rata-rata)
                    return sum(nums) / len(nums)
                return None
                
            # Copy data
            df_gaji = df.copy()
            
            # Buat kolom baru hasil pembersihan
            df_gaji['gaji_bersih'] = df_gaji['salary'].apply(bersihkan_gaji)
            
            # Buang baris "Not Available" 
            df_gaji = df_gaji.dropna(subset=['gaji_bersih'])
            
            # Cek jika masih ada data yang tersisa
            if not df_gaji.empty:
                gaji_summary = df_gaji.groupby('role_category')['gaji_bersih'].max().reset_index()
                gaji_summary = gaji_summary.sort_values(by='gaji_bersih', ascending=False)
                
                kolom_gaji_bar, kolom_gaji_box = st.columns(2)

                with kolom_gaji_bar:
                    fig_bar = px.bar(
                        gaji_summary, 
                        x='role_category', 
                        y='gaji_bersih',
                        title='Gaji Tertinggi per Kategori Role (Rp)', 
                        text_auto=True,
                        color='role_category', 
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_bar.update_layout(
                        template='plotly_dark', 
                        showlegend=False, 
                        height=400, 
                        yaxis_title="Gaji Tertinggi (Rp)", 
                        xaxis_title="Kategori Role"
                    )
                    # Mengganti fig_bar.show() menjadi format Streamlit
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with kolom_gaji_box:
                    fig_gaji_box = px.box(df_gaji, x='role_category', y='gaji_bersih',
                                          title='Rentang Distribusi Gaji per Role (Rp)',
                                          color='role_category', color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_gaji_box.update_layout(template='plotly_dark', showlegend=False, height=400,
                                               xaxis_title="Role Kategori", yaxis_title="Rentang Gaji")
                    st.plotly_chart(fig_gaji_box, use_container_width=True)
            else:
                st.warning("Semua baris pada kolom 'salary' bernilai 'Not Available' atau kosong.")
                
            st.markdown("---")

        # ANALISIS SKILL
        if 'job_description' in df.columns:
            st.subheader("🛠️ Top 10 Skill Paling Dicari")
            
            skills_to_track = [
                'excel', 'sql', 'python', 'dashboard', 'cloud', 
                'power bi', 'git', 'tableau', 'etl', 'statistics', 
                'machine learning', 'data warehouse','MLOps',
            ]
            skills_counter = Counter()

            # Loop untuk menghitung kemunculan keyword skill di job description
            for desc in df['job_description'].dropna():
                desc_lower = desc.lower()
                for skill in skills_to_track:
                    if skill in desc_lower:
                        skills_counter[skill] += 1

            top_skills = pd.DataFrame(skills_counter.most_common(10), columns=['Skill', 'Total Lowongan'])

            fig_skills = px.bar(top_skills, x='Total Lowongan', y='Skill', orientation='h', 
                                title='Top 10 Skill Paling Dicari Berdasarkan Deskripsi Kerja',
                                text_auto=True, color='Total Lowongan', color_continuous_scale='Plasma')
            
            fig_skills.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'},
                                     showlegend=False, height=400)
            
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.warning("Kolom 'job_description' tidak ditemukan. Analisis pencarian skill dilewati.")

else:
    st.info("Menunggu file diupload... Silakan drag-and-drop file dataset.")