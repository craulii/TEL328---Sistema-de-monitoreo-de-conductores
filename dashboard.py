import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path


st.set_page_config(
    page_title="Monitoreo de Conductores",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .fault-card {
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ddd;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .fault-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .critico {
        border-left: 5px solid #d62728;
    }
    .alto_riesgo {
        border-left: 5px solid #ff7f0e;
    }
    .sospechoso {
        border-left: 5px solid #ffdd57;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_database_connection():
    return sqlite3.connect('driver_monitoring_advanced.db', check_same_thread=False)


def load_faults():
    conn = get_database_connection()
    query = """
        SELECT id, timestamp, risk_level, fault_type, duration, 
               confidence, hands_on_wheel, person_count, objects_detected,
               image_path, video_clip_path
        FROM faults
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
    return df


def load_statistics():
    conn = get_database_connection()
    query = """
        SELECT id, session_start, session_end, total_frames, total_faults,
               critical_count, high_risk_count, suspicious_count, avg_fps
        FROM statistics
        ORDER BY session_start DESC
    """
    df = pd.read_sql_query(query, conn)
    if not df.empty:
        df['session_start'] = pd.to_datetime(df['session_start'])
        df['session_end'] = pd.to_datetime(df['session_end'])
        df['duration_minutes'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60
    return df


def overview_page():
    st.markdown('<p class="main-header">SISTEMA DE MONITOREO DE CONDUCTORES</p>', 
                unsafe_allow_html=True)
    
    faults_df = load_faults()
    stats_df = load_statistics()
    
    st.subheader("Metricas Generales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_faults = len(faults_df) if not faults_df.empty else 0
        st.metric("Total Faltas de Conductor", total_faults)
    
    with col2:
        critical = len(faults_df[faults_df['risk_level'] == 'CRITICO']) if not faults_df.empty else 0
        st.metric("Faltas Criticas", critical, delta_color="inverse")
    
    with col3:
        total_sessions = len(stats_df) if not stats_df.empty else 0
        st.metric("Sesiones Registradas", total_sessions)
    
    with col4:
        avg_fps = stats_df['avg_fps'].mean() if not stats_df.empty else 0
        st.metric("FPS Promedio", f"{avg_fps:.1f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not faults_df.empty:
            st.subheader("Distribucion por Nivel de Riesgo")
            risk_counts = faults_df['risk_level'].value_counts()
            colors = {'CRITICO': '#d62728', 'ALTO_RIESGO': '#ff7f0e', 'SOSPECHOSO': '#ffdd57'}
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de faltas de conductor disponibles")
    
    with col2:
        if not faults_df.empty:
            st.subheader("Top 5 Tipos de Faltas")
            fault_counts = faults_df['fault_type'].value_counts().head(5)
            fig = px.bar(
                x=fault_counts.values,
                y=[v.replace('_', ' ').title() for v in fault_counts.index],
                orientation='h',
                labels={'x': 'Cantidad', 'y': 'Tipo de Falta'},
                color=fault_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de faltas de conductor disponibles")
    
    if not faults_df.empty and len(faults_df) > 1:
        st.subheader("Timeline de Faltas de Conductor")
        faults_by_date = faults_df.groupby('date').size().reset_index(name='count')
        fig = px.line(
            faults_by_date,
            x='date',
            y='count',
            markers=True,
            labels={'date': 'Fecha', 'count': 'Numero de Faltas'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(height=350, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)


def faults_page():
    st.markdown('<p class="main-header">REGISTRO DE FALTAS DE CONDUCTOR</p>', 
                unsafe_allow_html=True)
    
    faults_df = load_faults()
    
    if faults_df.empty:
        st.warning("No se encontraron faltas de conductor registradas")
        return
    
    st.subheader("Filtros")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_filter = st.multiselect(
            "Nivel de Riesgo",
            options=faults_df['risk_level'].unique(),
            default=faults_df['risk_level'].unique()
        )
    
    with col2:
        fault_types = faults_df['fault_type'].unique()
        type_filter = st.multiselect(
            "Tipo de Falta",
            options=fault_types,
            default=fault_types
        )
    
    with col3:
        if not faults_df.empty:
            date_range = st.date_input(
                "Rango de Fechas",
                value=(faults_df['date'].min(), faults_df['date'].max()),
                min_value=faults_df['date'].min(),
                max_value=faults_df['date'].max()
            )
    
    with col4:
        view_mode = st.selectbox(
            "Modo de Vista",
            ["Tabla Interactiva", "Galeria de Tarjetas", "Lista Compacta"]
        )
    
    filtered_df = faults_df[
        (faults_df['risk_level'].isin(risk_filter)) &
        (faults_df['fault_type'].isin(type_filter))
    ]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) &
            (filtered_df['date'] <= date_range[1])
        ]
    
    st.divider()
    st.subheader(f"Resultados: {len(filtered_df)} faltas de conductor encontradas")
    
    if 'selected_fault_id' not in st.session_state:
        st.session_state.selected_fault_id = None
    
    if view_mode == "Tabla Interactiva":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Lista de Faltas de Conductor")
            
            for idx, row in filtered_df.iterrows():
                risk_class = row['risk_level'].lower().replace('_', '')
                
                button_label = f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {row['risk_level']} | {row['fault_type'].replace('_', ' ').title()}"
                
                if st.button(
                    button_label,
                    key=f"btn_{idx}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_fault_id == idx else "secondary"
                ):
                    st.session_state.selected_fault_id = idx
                    st.rerun()
        
        with col2:
            st.markdown("#### Detalles y Video")
            
            if st.session_state.selected_fault_id is not None:
                fault = filtered_df.loc[st.session_state.selected_fault_id]
                
                video_path = fault['video_clip_path']
                if pd.notna(video_path) and os.path.exists(video_path):
                    st.video(video_path, autoplay=True)
                else:
                    image_path = fault['image_path']
                    if pd.notna(image_path) and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
                    else:
                        st.info("No hay multimedia disponible")
                
                st.markdown("---")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"**Nivel:** :red[{fault['risk_level']}]")
                    st.markdown(f"**Duracion:** {fault['duration']:.1f}s")
                    st.markdown(f"**Manos en volante:** {fault['hands_on_wheel']}")
                
                with col_b:
                    st.markdown(f"**Confianza:** {fault['confidence']:.1%}")
                    st.markdown(f"**Personas:** {fault['person_count']}")
                    if pd.notna(fault['objects_detected']):
                        st.markdown(f"**Objetos:** {fault['objects_detected']}")
            else:
                st.info("Selecciona una falta de conductor de la lista para ver detalles")
    
    elif view_mode == "Galeria de Tarjetas":
        cards_per_row = 2
        
        for i in range(0, len(filtered_df), cards_per_row):
            cols = st.columns(cards_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(filtered_df):
                    row = filtered_df.iloc[i + j]
                    idx = filtered_df.index[i + j]
                    
                    with col:
                        with st.container():
                            risk_color = {'CRITICO': '🔴', 'ALTO_RIESGO': '🟠', 'SOSPECHOSO': '🟡'}
                            st.markdown(f"### {risk_color.get(row['risk_level'], '⚪')} {row['fault_type'].replace('_', ' ').title()}")
                            st.caption(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            video_path = row['video_clip_path']
                            if pd.notna(video_path) and os.path.exists(video_path):
                                st.video(video_path)
                            else:
                                image_path = row['image_path']
                                if pd.notna(image_path) and os.path.exists(image_path):
                                    st.image(image_path, use_container_width=True)
                            
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("Nivel", row['risk_level'])
                            with col_y:
                                st.metric("Duracion", f"{row['duration']:.1f}s")
                            with col_z:
                                st.metric("Confianza", f"{row['confidence']:.0%}")
                            
                            st.divider()
    
    else:
        display_df = filtered_df[['timestamp', 'risk_level', 'fault_type', 'duration', 'confidence']].copy()
        display_df.columns = ['Fecha/Hora', 'Nivel de Riesgo', 'Tipo', 'Duracion (s)', 'Confianza']
        display_df['Tipo'] = display_df['Tipo'].str.replace('_', ' ').str.title()
        display_df['Confianza'] = display_df['Confianza'].apply(lambda x: f"{x:.1%}")
        display_df['Duracion (s)'] = display_df['Duracion (s)'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        st.divider()
        
        fault_options = {
            f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {row['fault_type'].replace('_', ' ').title()}": idx
            for idx, row in filtered_df.iterrows()
        }
        
        selected = st.selectbox("Ver detalles de falta de conductor", options=list(fault_options.keys()))
        
        if selected:
            fault_idx = fault_options[selected]
            fault = filtered_df.loc[fault_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                video_path = fault['video_clip_path']
                if pd.notna(video_path) and os.path.exists(video_path):
                    st.video(video_path, autoplay=True)
                else:
                    image_path = fault['image_path']
                    if pd.notna(image_path) and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
            
            with col2:
                st.markdown("### Informacion Detallada")
                st.markdown(f"**Nivel:** :red[{fault['risk_level']}]")
                st.markdown(f"**Tipo:** {fault['fault_type'].replace('_', ' ').title()}")
                st.markdown(f"**Duracion:** {fault['duration']:.1f}s")
                st.markdown(f"**Confianza:** {fault['confidence']:.1%}")
                st.markdown(f"**Manos en volante:** {fault['hands_on_wheel']}")
                st.markdown(f"**Personas detectadas:** {fault['person_count']}")


def statistics_page():
    st.markdown('<p class="main-header">ESTADISTICAS DE RENDIMIENTO</p>', 
                unsafe_allow_html=True)
    
    stats_df = load_statistics()
    
    if stats_df.empty:
        st.warning("No se encontraron estadisticas de sesiones")
        return
    
    st.subheader("Metricas de Sesiones")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_frames = stats_df['total_frames'].sum()
        st.metric("Total Frames Procesados", f"{total_frames:,}")
    
    with col2:
        avg_fps = stats_df['avg_fps'].mean()
        st.metric("FPS Promedio Global", f"{avg_fps:.2f}")
    
    with col3:
        total_duration = stats_df['duration_minutes'].sum()
        st.metric("Tiempo Total Monitoreo", f"{total_duration:.0f} min")
    
    with col4:
        total_faults_all = stats_df['total_faults'].sum()
        st.metric("Faltas Totales", total_faults_all)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FPS por Sesion")
        fig = px.bar(
            stats_df,
            x=stats_df.index + 1,
            y='avg_fps',
            labels={'x': 'Sesion', 'avg_fps': 'FPS Promedio'},
            color='avg_fps',
            color_continuous_scale='Viridis'
        )
        fig.add_hline(y=avg_fps, line_dash="dash", line_color="red", 
                      annotation_text=f"Media: {avg_fps:.1f}")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribucion de Faltas por Nivel")
        fault_data = {
            'Nivel': ['Criticas', 'Alto Riesgo', 'Sospechosas'],
            'Cantidad': [
                stats_df['critical_count'].sum(),
                stats_df['high_risk_count'].sum(),
                stats_df['suspicious_count'].sum()
            ]
        }
        df_faults = pd.DataFrame(fault_data)
        colors_map = {'Criticas': '#d62728', 'Alto Riesgo': '#ff7f0e', 'Sospechosas': '#ffdd57'}
        fig = px.bar(
            df_faults,
            x='Nivel',
            y='Cantidad',
            color='Nivel',
            color_discrete_map=colors_map
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    if len(stats_df) > 1:
        st.subheader("Evolucion del Rendimiento")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats_df['session_start'],
            y=stats_df['avg_fps'],
            mode='lines+markers',
            name='FPS',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            xaxis_title='Fecha de Sesion',
            yaxis_title='FPS Promedio',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Historial de Sesiones")
    display_stats = stats_df[['session_start', 'duration_minutes', 'total_frames', 
                               'avg_fps', 'total_faults', 'critical_count']].copy()
    display_stats.columns = ['Inicio Sesion', 'Duracion (min)', 'Frames', 
                             'FPS Prom', 'Total Faltas', 'Criticas']
    display_stats['Inicio Sesion'] = display_stats['Inicio Sesion'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_stats['Duracion (min)'] = display_stats['Duracion (min)'].apply(lambda x: f"{x:.1f}")
    display_stats['FPS Prom'] = display_stats['FPS Prom'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_stats,
        use_container_width=True,
        height=400,
        hide_index=True
    )


def clips_gallery():
    st.markdown('<p class="main-header">GALERIA DE CLIPS</p>', 
                unsafe_allow_html=True)
    
    st.info("Selecciona una carpeta de nivel de riesgo")
    
    level_folders = {
        'Criticas': 'clips/critico',
        'Alto Riesgo': 'clips/altoriesgo',
        'Sospechosas': 'clips/sospechoso'
    }
    
    selected_level = st.selectbox("Nivel de Riesgo", options=list(level_folders.keys()))
    clips_dir = Path(level_folders[selected_level])
    
    if not clips_dir.exists():
        st.warning(f"La carpeta '{clips_dir}' no existe")
        return
    
    video_files = list(clips_dir.glob('*.mp4')) + list(clips_dir.glob('*.avi')) + list(clips_dir.glob('*.webm'))
    
    if not video_files:
        st.info(f"No hay clips de video en '{selected_level}'")
        return
    
    video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    st.subheader(f"Total de clips en {selected_level}: {len(video_files)}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        cols_per_row = st.slider("Videos por fila", 1, 4, 3)
    with col2:
        autoplay = st.checkbox("Reproduccion automatica", value=False)
    
    for i in range(0, len(video_files), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(video_files):
                video_file = video_files[i + j]
                with col:
                    file_date = datetime.fromtimestamp(video_file.stat().st_mtime)
                    file_size = video_file.stat().st_size / (1024 * 1024)
                    
                    st.markdown(f"**{video_file.name}**")
                    st.caption(f"{file_date.strftime('%Y-%m-%d %H:%M:%S')} | {file_size:.2f} MB")
                    
                    with open(video_file, 'rb') as f:
                        video_bytes = f.read()
                        st.video(video_bytes, autoplay=autoplay)
                    
                    st.divider()


def main():
    with st.sidebar:
        st.title("Navegacion")
        st.divider()
        
        page = st.radio(
            "Seleccionar pagina:",
            ["Resumen General", "Faltas de Conductor", "Estadisticas", "Galeria de Clips"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("### Informacion del Sistema")
        faults_df = load_faults()
        stats_df = load_statistics()
        
        if not faults_df.empty:
            last_fault = faults_df.iloc[0]['timestamp']
            st.info(f"Ultima falta:\n{last_fault.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not stats_df.empty:
            last_session = stats_df.iloc[0]['session_start']
            st.info(f"Ultima sesion:\n{last_session.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        if st.button("Actualizar Datos", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("Dashboard v2.0")
        st.caption("Sistema de Monitoreo de Conductores")
    
    if page == "Resumen General":
        overview_page()
    elif page == "Faltas de Conductor":
        faults_page()
    elif page == "Estadisticas":
        statistics_page()
    elif page == "Galeria de Clips":
        clips_gallery()


if __name__ == "__main__":
    main()
