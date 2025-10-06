import os
import re
import json
import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from zoneinfo import ZoneInfo

import folium
from folium import GeoJson, GeoJsonTooltip, LayerControl
from streamlit_folium import st_folium

import altair as alt
import streamlit.components.v1 as components

# =========================
# Config geral
# =========================
st.set_page_config(
    page_title="Perenização de Rios • Vazões", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Esconder a barra lateral completamente
st.markdown("""
    <style>
        .css-1d391kg {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .css-1lcbmhc {display: none;}
        header {visibility: hidden;}
        .css-1rs6os {visibility: hidden;}
        .css-17ziqus {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

TZ = ZoneInfo("America/Fortaleza")

# =========================
# Utilidades
# =========================
def normalize_col(s: str) -> str:
    if s is None:
        return ""
    t = s.strip().replace("\u200b", "")
    repl = {
        "á":"a","à":"a","ã":"a","â":"a","ä":"a",
        "é":"e","è":"e","ê":"e","ë":"e",
        "í":"i","ì":"i","î":"i","ï":"i",
        "ó":"o","ò":"o","õ":"o","ô":"o","ö":"o",
        "ú":"u","ù":"u","û":"u","ü":"u",
        "ç":"c",
        "Á":"a","À":"a","Ã":"a","Â":"a","Ä":"a",
        "É":"e","È":"e","Ê":"e","Ë":"e",
        "Í":"i","Ì":"i","Î":"i","Ï":"i",
        "Ó":"o","Ò":"o","Õ":"o","Ô":"o","Ö":"o",
        "Ú":"u","Ù":"u","Û":"u","Ü":"u",
        "Ç":"c",
    }
    for a,b in repl.items(): t = t.replace(a,b)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t

@st.cache_data(show_spinner=False)
def load_from_gsheet_csv(sheet_id: str, gid: str = "0", sep=","):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url, sep=sep)

def guess_columns(df: pd.DataFrame):
    aliases = {
        "data": ["data", "data de medicao", "data da medicao", "dt", "data_medicao"],
        "responsavel": ["responsavel", "responsável", "📃 responsavel pela informacao", "📃 responsavel pela informação"],
        "campanha": ["campanha", "🔘 campanha"],
        "reservatorio": ["reservatorio/sistema", "reservatorio", "reservatório/sistema", "💧 selecione o reservatorio/sistema"],
        "secao": ["secao", "seção", "📍 selecione a secao", "selecione a secao", "selecione a seção"],
        "vazao": ["vazao medida", "vazão medida", "vaz_o_medida"],
        "obs": ["observacoes", "observações", "✏️ observacoes", "✏️ observações", "_observa_es"],
        "lat": ["latitude", "__coordenadas_latitude", "lat"],
        "lon": ["longitude", "__coordenadas_longitude", "lng", "long", "lon"],
        "foto1": ["foto do local_url", "📷 foto do local_url", "foto 01_url", "foto (01)_url"],
        "foto2": ["foto (02)_url", "foto 02_url"],
        "foto3": ["foto (03)_url", "foto 03_url"],
        "video": ["video do local_url", "vídeo do local_url", "video_url"],
    }
    norm_map = {c: normalize_col(c) for c in df.columns}
    reverse = {}
    for orig, norm in norm_map.items():
        reverse.setdefault(norm, orig)

    found = {}
    for key, opts in aliases.items():
        picked = None
        for cand in opts:
            norm_cand = normalize_col(cand)
            if norm_cand in reverse:
                picked = reverse[norm_cand]
                break
        if not picked:
            for norm_col, orig_col in reverse.items():
                if any(norm_col.startswith(normalize_col(o)) for o in opts):
                    picked = orig_col
                    break
        if picked:
            found[key] = picked
    return found

# --------- Google Drive helpers ----------
def gdrive_extract_id(url: str):
    if not isinstance(url, str):
        return None
    url = url.strip()
    m = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if m: return m.group(1)
    return None

def drive_image_urls(file_id: str):
    """Thumb e imagem grande (ambas image/*)."""
    thumb = f"https://drive.google.com/thumbnail?id={file_id}&sz=w480"
    big   = f"https://drive.google.com/thumbnail?id={file_id}&sz=w2048"
    return thumb, big

def drive_video_embed(file_id: str):
    """Preview em iframe para vídeo do Drive."""
    return f"https://drive.google.com/file/d/{file_id}/preview"

def render_lightgallery_mixed(items: list, height_px=440):
    """
    items = [{ 'thumb':..., 'src':..., 'caption':..., 'iframe': bool }]
    """
    if not items:
        st.info("Sem mídias para exibir.")
        return

    anchors = []
    for it in items:
        if it.get("iframe"):
            anchors.append(
                f'''<a class="gallery-item" data-iframe="true" data-src="{it["src"]}" data-sub-html="{it.get("caption","")}">
                        <img src="{it["thumb"]}" loading="lazy"/>
                    </a>'''
            )
        else:
            anchors.append(
                f'''<a class="gallery-item" href="{it["src"]}" data-sub-html="{it.get("caption","")}">
                        <img src="{it["thumb"]}" loading="lazy"/>
                    </a>'''
            )

    items_html = "\n".join(anchors)

    html = f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/css/lightgallery-bundle.min.css">
    <style>
      .lg-backdrop {{ background: rgba(0,0,0,0.9); }}
      .gallery-container {{ display:flex; flex-wrap: wrap; gap: 12px; align-items:flex-start; }}
      .gallery-item img {{ 
          height: 140px; 
          width: auto; 
          border-radius: 10px; 
          box-shadow: 0 4px 12px rgba(0,0,0,.2);
          transition: transform 0.3s ease, box-shadow 0.3s ease;
      }}
      .gallery-item:hover img {{
          transform: scale(1.05);
          box-shadow: 0 6px 16px rgba(0,0,0,.3);
      }}
    </style>

    <div id="lg-mixed" class="gallery-container">{items_html}</div>

    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/lightgallery.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/plugins/zoom/lg-zoom.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/plugins/thumbnail/lg-thumbnail.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/plugins/video/lg-video.umd.js"></script>

    <script>
      window.addEventListener('load', () => {{
        const container = document.getElementById('lg-mixed');
        lightGallery(container, {{
          selector: '.gallery-item',
          zoom: true,
          thumbnail: true,
          download: false,
          controls: true,
          loop: true,
          plugins: [lgZoom, lgThumbnail, lgVideo]
        }});
      }});
    </script>
    """
    components.html(html, height=height_px, scrolling=True)

def make_popup_html(row, cols):
    safe = lambda x: "-" if (x is None or (isinstance(x,float) and math.isnan(x))) else str(x)
    labels = {
        "campanha": "Campanha",
        "reservatorio": "Reservatório/Sistema",
        "secao": "Seção",
        "vazao": "Vazão medida",
    }
    
    # HTML moderno para o popup
    parts = []
    for k in ["campanha", "reservatorio", "secao", "vazao"]:
        colname = cols.get(k)
        if colname and colname in row and pd.notna(row[colname]):
            value = safe(row[colname])
            if k == "vazao":
                # Destacar a vazão
                value = f'<span style="color: #e74c3c; font-weight: bold; font-size: 1.1em;">{value}</span>'
            parts.append(f'<div class="popup-item"><span class="popup-label">{labels[k]}:</span> <span class="popup-value">{value}</span></div>')
    
    popup_html = f"""
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 12px;
        min-width: 220px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        color: white;
        border: 2px solid rgba(255,255,255,0.1);
    ">
        <div style="
            background: rgba(255,255,255,0.1); 
            padding: 8px 12px; 
            border-radius: 8px; 
            margin-bottom: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        ">
            <strong>📊 Dados da Medição</strong>
        </div>
        {'<hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">'.join(parts)}
        <div style="
            margin-top: 10px;
            padding: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
            text-align: center;
            font-size: 0.85em;
            backdrop-filter: blur(5px);
        ">
            🗺️ Clique para mais detalhes
        </div>
    </div>
    """
    return popup_html

def load_geojson_safe(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def geojson_bounds(gj: dict):
    """
    Calcula bounds [ [min_lat,min_lon], [max_lat,max_lon] ] de um GeoJSON.
    """
    if not gj: 
        return None

    def walk_coords(coords):
        # retorna lista de (lon, lat)
        if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
            # coord única [lon, lat, ...]
            return [(coords[0], coords[1])]
        result = []
        for c in coords:
            result.extend(walk_coords(c))
        return result

    geoms = []
    if gj.get("type") == "FeatureCollection":
        geoms = [f.get("geometry") for f in gj.get("features", []) if f.get("geometry")]
    elif gj.get("type") == "Feature":
        geoms = [gj.get("geometry")]
    else:
        geoms = [gj]

    min_lon, min_lat = 180.0, 90.0
    max_lon, max_lat = -180.0, -90.0

    for geom in geoms:
        if not geom: 
            continue
        coords = geom.get("coordinates")
        if coords is None:
            continue
        for lon, lat in walk_coords(coords):
            if lon is None or lat is None:
                continue
            min_lon = min(min_lon, float(lon)); min_lat = min(min_lat, float(lat))
            max_lon = max(max_lon, float(lon)); max_lat = max(max_lat, float(lat))

    if min_lon == 180.0:
        return None
    return [[min_lat, min_lon], [max_lat, max_lon]]  # folium usa [lat, lon]

# Caminhos possíveis (local e /mnt/data/)
TRECHOS_CAND = ["trechos_perene.geojson", "/mnt/data/trechos_perene.geojson"]
BACIA_CAND   = ["bacia_banabuiu.geojson", "/mnt/data/bacia_banabuiu.geojson",
                "bacia_banabuiu.geojason", "/mnt/data/bacia_banabuiu.geojason"]

# =========================
# App
# =========================
def main():
    # Header moderno
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 0 0 20px 20px;
            margin: -1rem -1rem 2rem -1rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        ">
            <h1 style="margin:0; font-size: 2.5rem; font-weight: 700;">🌊 Monitoramento de Vazões</h1>
            <p style="margin:0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Perenização de Rios • Sistema de Análise</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"🕐 Atualizado em {datetime.now(TZ).strftime('%d/%m/%Y %H:%M:%S')} — Fuso America/Fortaleza")

    # Carregamento automático do Google Sheets (configuração simplificada)
    SHEET_ID = "1YstNFY5ehrOUjg_AoSztcqq466uRwstKY7gpvs0BWnI"
    GID = "0"
    SEP = ","

    try:
        df = load_from_gsheet_csv(SHEET_ID, GID, sep=SEP)
    except Exception as e:
        st.error(f"❌ Erro ao carregar do Google Sheets: {e}")
        return

    if df.empty:
        st.info("📭 Sem dados. Verifique permissões do Sheets e o GID informado.")
        return

    # Normaliza nulos e descobre colunas
    df = df.replace({np.nan: None})
    cols = guess_columns(df)

    # Data
    if cols.get("data") and cols["data"] in df.columns:
        df[cols["data"]] = pd.to_datetime(df[cols["data"]], errors="coerce", dayfirst=True)

    # =========================
    # FILTROS
    # =========================
    st.markdown("---")
    st.subheader("🔍 Filtros de Dados")

    # Container moderno para filtros
    with st.container():
        if cols.get("data"):
            min_d = pd.to_datetime(df[cols["data"]]).min()
            max_d = pd.to_datetime(df[cols["data"]]).max()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                data_ini = st.date_input("**Data inicial**", 
                                       value=min_d.date() if pd.notna(min_d) else date.today(),
                                       help="Data inicial do período")
            with col2:
                data_fim = st.date_input("**Data final**", 
                                       value=max_d.date() if pd.notna(max_d) else date.today(),
                                       help="Data final do período")
        else:
            col1, col2, col3, col4 = st.columns(4)
            data_ini = data_fim = None
            st.warning("⚠️ Coluna de **Data** não identificada automaticamente.")

        def options_for(colkey):
            cname = cols.get(colkey)
            if not cname or cname not in df.columns:
                return []
            vals = sorted({v for v in df[cname].dropna().tolist()})
            return vals

        with col3:
            camp = st.multiselect("**Campanha**", options_for("campanha"), help="Filtrar por campanha")
        with col4:
            rese = st.multiselect("**Reservatório/Sistema**", options_for("reservatorio"), help="Filtrar por reservatório")
        
        secao_opts = options_for("secao")
        sec_sel = st.multiselect("**Seção**", secao_opts, help="Filtrar por seção específica")

    # Aplicar filtros
    fdf = df.copy()
    if cols.get("data") and (data_ini and data_fim):
        mask = (fdf[cols["data"]].dt.date >= data_ini) & (fdf[cols["data"]].dt.date <= data_fim)
        fdf = fdf.loc[mask]

    def filt_in(colkey, selected):
        cname = cols.get(colkey)
        if not cname or not selected:
            return
        fdf.loc[:, cname] = fdf[cname].astype(str)
        sel = set(map(str, selected))
        return fdf[cname].isin(sel)

    for key, selected in [("campanha", camp), ("reservatorio", rese), ("secao", sec_sel)]:
        flt = filt_in(key, selected)
        if flt is not None:
            fdf = fdf.loc[flt]

    # Badge de resultado
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        ">
            📈 Registros após filtros: <strong>{len(fdf)}</strong>
        </div>
    """, unsafe_allow_html=True)

    # =========================
    # TABELA + MÍDIAS (seletor)
    # =========================
    st.markdown("---")
    st.subheader("📋 Dados e Mídias")
    col_tab, col_media = st.columns([1, 1])

    with col_tab:
        st.markdown("**📊 Tabela de Registros**")
        table_cols = [
            cols.get("campanha"),
            cols.get("reservatorio"),
            cols.get("secao"),
            cols.get("vazao"),
            cols.get("obs"),
        ]
        table_cols = [c for c in table_cols if c in fdf.columns and c is not None]
        if table_cols:
            st.dataframe(
                fdf[table_cols].rename(columns={
                    cols.get("campanha",""): "Campanha",
                    cols.get("reservatorio",""): "Reservatório/Sistema",
                    cols.get("secao",""): "Seção",
                    cols.get("vazao",""): "Vazão medida",
                    cols.get("obs",""): "Observações",
                }),
                use_container_width=True,
                height=420
            )
        else:
            st.warning("⚠️ Não encontrei as colunas necessárias para a tabela solicitada.")

    with col_media:
        st.markdown("**🖼️ Galeria de Mídias**")

        media_map = {
            "Foto do local_URL": cols.get("foto1"),
            "Foto (02)_URL":    cols.get("foto2"),
            "Foto (03)_URL":    cols.get("foto3"),
            "Video do Local_URL": cols.get("video"),
        }
        valid_options = [label for label, cname in media_map.items() if cname and cname in fdf.columns]
        if not valid_options:
            st.info("📭 Nenhuma coluna de mídia encontrada na base.")
        else:
            choice = st.selectbox("**Selecione o tipo de mídia**", valid_options, index=0)

            def split_urls(cell: str):
                parts = re.split(r"[,\n; ]+", cell.strip())
                return [p.strip() for p in parts if p.strip().lower().startswith(("http://","https://"))]

            items = []
            seen = set()
            cname = media_map[choice]

            for _, row in fdf.iterrows():
                cell = row.get(cname)
                if not isinstance(cell, str): 
                    continue

                # Caption: Reservatório/Sistema • Seção
                rlab = row.get(cols.get("reservatorio",""))
                slab = row.get(cols.get("secao",""))
                caption = " • ".join([x for x in [str(rlab) if rlab else None, str(slab) if slab else None] if x])

                for u in split_urls(cell):
                    if u in seen:
                        continue
                    seen.add(u)

                    if "Video" in choice:
                        fid = gdrive_extract_id(u)
                        if fid:
                            thumb, _ = drive_image_urls(fid)
                            src = drive_video_embed(fid)
                            items.append({"thumb": thumb, "src": src, "caption": caption, "iframe": True})
                        else:
                            items.append({"thumb": u, "src": u, "caption": caption, "iframe": True})
                    else:
                        fid = gdrive_extract_id(u)
                        if fid:
                            thumb, big = drive_image_urls(fid)
                            items.append({"thumb": thumb, "src": big, "caption": caption, "iframe": False})
                        else:
                            items.append({"thumb": u, "src": u, "caption": caption, "iframe": False})

            if items:
                render_lightgallery_mixed(items, height_px=420)
            else:
                st.info("📭 Sem mídias para exibir nessa coluna. Verifique se os links apontam para arquivos do Drive (não pastas) e se estão compartilhados como 'qualquer pessoa com o link'.")

    # =========================
    # MAPA — Folium (wide)
    # =========================
    st.markdown("---")
    st.subheader("🗺️ Mapa das Seções Monitoradas")
    map_height = 520

    # Mapa base com estilo moderno
    fmap = folium.Map(
        location=[-5.199, -39.292], 
        zoom_start=8, 
        control_scale=True, 
        prefer_canvas=True,
        tiles='CartoDB Positron'
    )

    # Tiles (bases) com attribution
    folium.TileLayer("CartoDB Positron", name="🗺️ CartoDB Positron").add_to(fmap)
    folium.TileLayer("CartoDB Dark_Matter", name="🌙 CartoDB Dark Matter").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="🌍 OpenStreetMap").add_to(fmap)
  
    # FeatureGroups para poder ligar/desligar
    fg_bacia   = folium.FeatureGroup(name="🏞️ Bacia do Banabuiú", show=True)
    fg_trechos = folium.FeatureGroup(name="🌊 Trechos Perene", show=True)
    fg_pontos  = folium.FeatureGroup(name="📍 Pontos de Medição", show=True)

    # GeoJSON camadas
    trechos = load_geojson_safe(*TRECHOS_CAND)
    bacia   = load_geojson_safe(*BACIA_CAND)

    if trechos:
        GeoJson(
            trechos,
            name="Trechos Perene",
            style_function=lambda x: {
                "color": "#3498db",
                "weight": 4,
                "opacity": 0.9,
                "dashArray": "5, 5"
            },
            tooltip=GeoJsonTooltip(fields=[], aliases=[], sticky=False)
        ).add_to(fg_trechos)
        fg_trechos.add_to(fmap)

    bacia_bounds = None
    if bacia:
        GeoJson(
            bacia,
            name="Bacia do Banabuiú",
            style_function=lambda x: {
                "color": "#27ae60",
                "weight": 3,
                "opacity": 0.8, 
                "fillOpacity": 0.05,
                "fillColor": "#27ae60"
            }
        ).add_to(fg_bacia)
        fg_bacia.add_to(fmap)
        bacia_bounds = geojson_bounds(bacia)

    # Pontos de medição
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    pts = []
    if lat_col and lon_col and lat_col in fdf.columns and lon_col in fdf.columns:
        def to_float(v):
            if v is None: return None
            if isinstance(v, str): v = v.replace(",", ".")
            try: return float(v)
            except: return None

        for _, row in fdf.iterrows():
            lat = to_float(row.get(lat_col))
            lon = to_float(row.get(lon_col))
            if lat is None or lon is None:
                continue
            popup_html = make_popup_html(row, cols)
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="#e74c3c",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300, parse_html=True),
                tooltip=f"📍 {str(row.get(cols.get('secao',''), 'Seção'))}",
            ).add_to(fg_pontos)
            pts.append((lat, lon))
        fg_pontos.add_to(fmap)

    # ---- FIT: prioriza a Bacia; se ausente, usa pontos; senão mantém default
    if bacia_bounds:
        fmap.fit_bounds(bacia_bounds)
    elif pts:
        fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                         [max(p[0] for p in pts), max(p[1] for p in pts)]])

    # Botão de seleção de camadas (colapsado = aparece como botão)
    LayerControl(collapsed=True, position="topright").add_to(fmap)

    st_folium(fmap, height=map_height, use_container_width=True)

    # =========================
    # GRÁFICOS (Data, Seção, Vazão medida)
    # =========================
    st.markdown("---")
    st.subheader("📈 Análises Gráficas")
    
    if cols.get("data") and cols.get("secao") and cols.get("vazao"):
        gdf = fdf[[cols["data"], cols["secao"], cols["vazao"]]].dropna()
        gdf[cols["vazao"]] = pd.to_numeric(gdf[cols["vazao"]].astype(str).str.replace(",", "."), errors="coerce")
        gdf = gdf.dropna(subset=[cols["vazao"]])

        # Gráfico de linha
        st.markdown("**📈 Vazão ao Longo do Tempo**")
        line = alt.Chart(gdf).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X(f"{cols['data']}:T", title="Data", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida"),
            color=alt.Color(f"{cols['secao']}:N", title="Seção", 
                          scale=alt.Scale(scheme='category10')),
            tooltip=[cols["data"], cols["secao"], cols["vazao"]]
        ).properties(width="container", height=400)
        st.altair_chart(line, use_container_width=True)

        # Gráfico de boxplot
        st.markdown("**📊 Distribuição de Vazão por Seção**")
        box = alt.Chart(gdf).mark_boxplot(size=30).encode(
            x=alt.X(f"{cols['secao']}:N", title="Seção", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida"),
            color=alt.Color(f"{cols['secao']}:N", legend=None,
                          scale=alt.Scale(scheme='category10'))
        ).properties(width="container", height=400)
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("📊 Para os gráficos, são necessárias as colunas **Data**, **Seção** e **Vazão medida**.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>© 2024 Sistema de Monitoramento de Vazões • Desenvolvido com Python 🐍</p>
            <p style="font-size: 0.9em;">Streamlit + Folium + Altair + LightGallery</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
