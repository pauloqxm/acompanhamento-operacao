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
from folium.features import CustomIcon
from streamlit_folium import st_folium

import altair as alt
import streamlit.components.v1 as components
from branca.element import IFrame  # popups HTML

# =========================
# Config geral
# =========================
st.set_page_config(
    page_title="Acompanhamento da Operação 2025.2",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Esconde sidebar/header (visual limpo)
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

# --- Estilização extra: botão primário como pílula verde, igual ao badge ---
st.markdown("""
    <style>
        /* Botão primário com gradiente verde (pílula) */
        .stButton > button[kind="primary"],
        div[data-testid="baseButton-primary"] > button {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.6rem 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
            transition: transform .05s ease-in-out, box-shadow .2s ease-in-out !important;
        }
        .stButton > button[kind="primary"]:hover,
        div[data-testid="baseButton-primary"] > button:hover {
            filter: brightness(1.03);
            box-shadow: 0 6px 16px rgba(0,0,0,0.18) !important;
        }
        .stButton > button[kind="primary"]:active,
        div[data-testid="baseButton-primary"] > button:active {
            transform: scale(0.99);
        }
    </style>
""", unsafe_allow_html=True)

TZ = ZoneInfo("America/Fortaleza")

# =========================
# Parâmetros FIXOS (sem controles)
# =========================
ICON_URL = "https://i.ibb.co/FqW4BJZr/rio-meidcao.png"
USE_ICON = True
ICON_W, ICON_H = 30, 30

POPUP_WIDTH = 360
POPUP_HEIGHT = 300

MAP_HEIGHT = 520

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
def load_from_gsheet_csv(sheet_id: str, gid: str = "0", sep=",", _bust: int = 0):
    # _bust entra na URL para evitar cache intermediário do Google/CDN
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}&_cachebust={_bust}"
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
    thumb = f"https://drive.google.com/thumbnail?id={file_id}&sz=w450"
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

# =========================================================================================
# POPUP estilizado
# =========================================================================================
def make_popup_html(row, cols):
    safe = lambda x: "-" if (x is None or (isinstance(x,float) and math.isnan(x))) else str(x)
    labels = {
        "data": "Data da Medição",
        "campanha": "Campanha",
        "reservatorio": "Reservatório/Sistema",
        "secao": "Seção",
        "vazao": "Vazão medida",
    }
    icons = {"data":"📅","campanha":"🏷️","reservatorio":"💧","secao":"📍","vazao":"🌊"}

    def split_urls(cell: str):
        if not isinstance(cell, str):
            return []
        parts = re.split(r"[,\n; ]+", cell.strip())
        return [p.strip() for p in parts if p.strip().lower().startswith(("http://", "https://"))]

    def build_img_thumb_big(url: str):
        fid = gdrive_extract_id(url)
        if fid:
            t, b = drive_image_urls(fid)
            is_video = "/preview" in url or url.lower().endswith((".mp4", ".mov", ".webm"))
            return (t, drive_video_embed(fid) if is_video else b, is_video)
        is_video = url.lower().endswith((".mp4", ".mov", ".webm"))
        return (url, url, is_video)

    date_col = cols.get("data")
    data_part = ''
    if date_col and date_col in row and pd.notna(row[date_col]):
        try:
            data_medicao = row[date_col].strftime('%d/%m/%Y')
            data_part = f'''<div style="display:flex;justify-content:space-between;padding-bottom:5px;font-size:0.9em;border-bottom:1px solid rgba(255,255,255,0.3);"><span>{icons['data']} {labels['data']}:</span><span id="med-data" style="font-weight:bold;color:#f1c40f;">{data_medicao}</span></div><div style="height:1px;background-color:rgba(255,255,255,0.2);margin:6px 0;"></div>'''
        except Exception:
            pass

    parts = []
    for k in ["campanha", "reservatorio", "secao", "vazao"]:
        colname = cols.get(k)
        if colname and colname in row and pd.notna(row[colname]):
            value = safe(row[colname])
            label = labels[k]
            icon = icons[k]
            if k == "vazao":
                try:
                    vazao_f = float(str(value).replace(',', '.'))
                    formatted_vazao = f"{vazao_f:,.2f} L/s".replace('.', '#').replace(',', '.').replace('#', ',')
                    value = f'<span id="med-vazao" style="color:#FF5733;font-weight:700;font-size:1.2em;">{formatted_vazao}</span>'
                except ValueError:
                    value = f'<span id="med-vazao" style="color:#FF5733;font-weight:700;">{value} L/s</span>'
            parts.append(f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:0.95em;"><span style="font-weight:500;">{icon} {label}:</span><span style="font-weight:bold;text-align:right;">{value}</span></div>')

    content_html = "".join(parts)

    # seletor de data
    selector_html = ''
    try:
        sec_col = cols.get('secao')
        dat_col = cols.get('data')
        vaz_col = cols.get('vazao')
        if sec_col and dat_col and vaz_col:
            sec_val = row.get(sec_col)
            from inspect import currentframe
            frame = currentframe()
            fdf_local = None
            while frame:
                if 'fdf' in frame.f_locals:
                    fdf_local = frame.f_locals['fdf']
                    break
                frame = frame.f_back
            if fdf_local is not None:
                tmp = fdf_local.loc[fdf_local[sec_col] == sec_val, [dat_col, vaz_col]].dropna()
                tmp = tmp.sort_values(by=dat_col, ascending=True)
                if len(tmp) > 1:
                    opts = []
                    for d, v in tmp.itertuples(index=False, name=None):
                        try:
                            dstr = pd.to_datetime(d, errors='coerce').strftime('%d/%m/%Y')
                        except Exception:
                            dstr = str(d)
                        try:
                            vf = float(str(v).replace(',', '.'))
                            vfmt = f"{vf:,.2f} L/s".replace('.', '#').replace(',', '.').replace('#', ',')
                        except Exception:
                            vfmt = f"{v} L/s"
                        sel = ' selected' if 'data_medicao' in locals() and dstr == data_medicao else ''
                        opts.append(f"<option value='{dstr}|{vfmt}'{sel}>{dstr}</option>")
                    sel_id = f"sel-{abs(hash(str(sec_val)))%10**8}"
                    selector_html = f"""
                    <div style='margin:6px 0 8px 0;'>
                        <label style='font-size:0.85em;opacity:.95;margin-right:6px;color:#fff;'>Alterar data:</label>
                        <select id='{sel_id}' style='padding:6px 10px;border-radius:6px;border:1px solid rgba(255,255,255,.35);background:rgba(0,0,0,0.3);color:#fff;font-size:0.9em;'>
                            {''.join(opts)}
                        </select>
                    </div>
                    <script>
                        (function(){{
                            var el = document.getElementById('{sel_id}');
                            if(!el) return;
                            el.addEventListener('change', function(){{
                                var parts = this.value.split('|');
                                var d = parts[0];
                                var v = parts.slice(1).join('|');
                                var dSpan = document.getElementById('med-data');
                                var vSpan = document.getElementById('med-vazao');
                                if(dSpan) dSpan.textContent = d;
                                if(vSpan) vSpan.innerHTML = v;
                            }});
                        }})();
                    </script>
                    """
    except Exception:
        selector_html = ''

    content_html = content_html + selector_html

    thumb_items = []
    rlab = row.get(cols.get("reservatorio", ""))
    slab = row.get(cols.get("secao", ""))
    caption = " • ".join([x for x in [str(rlab) if rlab else None, str(slab) if slab else None] if x])

    for k in ["foto1", "foto2", "foto3"]:
        cname = cols.get(k)
        if not cname or cname not in row:
            continue
        for u in split_urls(row.get(cname)):
            t, b, is_video = build_img_thumb_big(u)
            if is_video:
                continue
            thumb_items.append((t, b))

    thumbs_html = ''
    if thumb_items:
        cards = []
        for (t, b) in thumb_items:
            cards.append(f'<a href="{b}" target="_blank" title="Clique para ampliar"><img src="{t}" alt="{caption}" style="height:64px;width:auto;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.25);margin:4px;border:1px solid rgba(255,255,255,.35)"/></a>')
        thumbs_html = f'<div style="margin-top:10px"><div style="font-size:0.9em;margin-bottom:6px;opacity:.95">📷 Miniaturas</div><div style="display:flex;flex-wrap:wrap;align-items:center;">{''.join(cards)}</div></div>'

    popup_html = f"""<div style='font-family:Segoe UI, Tahoma, Geneva, Verdana, sans-serif;padding:15px;min-width:250px;max-width:350px;background:linear-gradient(135deg,#1abc9c 0%,#3498db 100%);border-radius:15px;box-shadow:0 10px 30px rgba(0,0,0,0.3);color:white;border:3px solid rgba(255,255,255,0.2);'><div style='background:rgba(255,255,255,0.15);padding:10px 15px;border-radius:10px;margin-bottom:15px;text-align:center;font-size:1.1em;font-weight:bold;letter-spacing:0.5px;text-shadow:1px 1px 2px rgba(0,0,0,0.2);'>Informações da Medição</div>{data_part}{content_html}{thumbs_html}<div style='margin-top:12px;padding:8px;background:rgba(255,255,255,0.1);border-radius:8px;text-align:center;font-size:0.8em;opacity:0.9;font-style:italic;'>Clique nas miniaturas para ampliar em nova aba.</div></div>"""
    return popup_html

# =========================================================================================

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
    """Bounds [[min_lat,min_lon],[max_lat,max_lon]] de um GeoJSON."""
    if not gj:
        return None

    def walk_coords(coords):
        if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
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
    return [[min_lat, min_lon], [max_lat, max_lon]]

# Caminhos possíveis (local e /mnt/data/)
TRECHOS_CAND = ["trechos_perene.geojson", "/mnt/data/trechos_perene.geojson"]
BACIA_CAND   = ["bacia_banabuiu.geojson", "/mnt/data/bacia_banabuiu.geojson",
                "bacia_banabuiu.geojason", "/mnt/data/bacia_banabuiu.geojason"]

# =========================
# App
# =========================
def main():
    # Estado para cache busting e botão de atualização
    if "cache_bust" not in st.session_state:
        st.session_state["cache_bust"] = 0

    # Header (largura total)
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
            padding: 2rem;border-radius: 0 0 20px 20px;margin: -1rem -1rem 2rem -1rem;
            color: white;text-align: center;box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        ">
            <h1 style="margin:0;font-size: 2.5rem;font-weight:700;">🌊Acompanhamento da Operação 2025.2</h1>
            <p style="margin:0.5rem 0 0 0;font-size:1.2rem;opacity: 0.9;">Perenização de Rios • Sistema de Análise de Dados</p>
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"🕐 Última atualização dos dados: {datetime.now(TZ).strftime('%d/%m/%Y %H:%M:%S')} — Fuso America/Fortaleza")

    # Google Sheets
    SHEET_ID = "1YstNFY5ehrOUjg_AoSztcqq466uRwstKY7gpvs0BWnI"
    GID = "0"
    SEP = ","

    try:
        df = load_from_gsheet_csv(SHEET_ID, GID, sep=SEP, _bust=st.session_state.get("cache_bust", 0))
    except Exception as e:
        st.error(f"❌ Erro ao carregar do Google Sheets: {e}")
        return

    if df.empty:
        st.info("📭 Sem dados. Verifique permissões do Sheets e o GID informado.")
        return

    df = df.replace({np.nan: None})
    cols = guess_columns(df)

    if cols.get("data") and cols["data"] in df.columns:
        df[cols["data"]] = pd.to_datetime(df[cols["data"]], errors="coerce", dayfirst=True)

    # =========================
    # FILTROS
    # =========================
    st.markdown("---")
    st.subheader("🔍 Filtros de Dados")

    with st.container():
        if cols.get("data"):
            valid_dates = pd.to_datetime(df[cols["data"]]).dropna()
            min_d = valid_dates.min() if not valid_dates.empty else date.today()
            max_d = valid_dates.max() if not valid_dates.empty else date.today()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                data_ini = st.date_input("**Data inicial**", value=min_d.date() if pd.notna(min_d) else date.today())
            with col2:
                data_fim = st.date_input("**Data final**", value=max_d.date() if pd.notna(max_d) else date.today())
        else:
            col1, col2, col3, col4 = st.columns(4)
            data_ini = data_fim = None
            st.warning("⚠️ Coluna de **Data** não identificada automaticamente.")

        def options_for(colkey):
            cname = cols.get(colkey)
            if not cname or cname not in df.columns:
                return []
            vals = sorted({v for v in df[cname].dropna().astype(str).tolist()})
            return vals

        with col3:
            camp = st.multiselect("**Campanha**", options_for("campanha"))
        with col4:
            rese = st.multiselect("**Reservatório/Sistema**", options_for("reservatorio"))

        secao_opts = options_for("secao")
        sec_sel = st.multiselect("**Seção**", secao_opts)

    # Aplicar filtros
    fdf = df.copy()
    if cols.get("data") and (data_ini and data_fim):
        date_series = pd.to_datetime(fdf[cols["data"]], errors='coerce')
        mask = (date_series.dt.date >= data_ini) & (date_series.dt.date <= data_fim)
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

    # =========================
    # MÉTRICA + BOTÃO (lado a lado)
    # =========================
    metric_l, metric_r = st.columns([1, 0.25])
    with metric_l:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
                color: white;padding: 0.5rem 1rem;border-radius: 20px;display: inline-block;
                margin: 0.5rem 0;font-weight: bold;box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            ">
                📈 Registros após filtros: <strong>{len(fdf)}</strong>
            </div>
        """, unsafe_allow_html=True)
    with metric_r:
        if st.button("🔄 Atualizar dados do Sheets", use_container_width=True, type="primary"):
            load_from_gsheet_csv.clear()
            st.session_state["cache_bust"] += 1
            st.rerun()

    # =========================
    # TABELA + MÍDIAS
    # =========================
    st.markdown("---")
    st.subheader("📋 Dados e Mídias")
    col_tab, col_media = st.columns([1, 1])

    # ---------- Tabela ----------
    with col_tab:
        st.markdown("**📊 Tabela de Registros**")

        display_df = fdf.copy()

        # Garante formato dd/mm/aaaa
        if cols.get("data") and cols["data"] in display_df.columns:
            display_df["Data formatada"] = pd.to_datetime(display_df[cols["data"]], errors="coerce").dt.strftime("%d/%m/%Y")
        else:
            display_df["Data formatada"] = None

        # Remove a primeira coluna se for puramente numérica (apenas na exibição)
        if display_df.columns.size > 0:
            first_col = display_df.columns[0]
            try:
                if display_df[first_col].dropna().astype(str).str.isnumeric().all():
                    display_df = display_df.drop(columns=first_col)
            except Exception:
                pass

        # Define as colunas a exibir na tabela
        table_cols = [
            "Data formatada",
            cols.get("campanha"),
            cols.get("reservatorio"),
            cols.get("secao"),
            cols.get("vazao"),
            cols.get("obs"),
        ]
        table_cols = [c for c in table_cols if c in display_df.columns and c is not None]

        if table_cols:
            renamed = {
                "Data formatada": "Data da Medição",
                cols.get("campanha"," "): "Campanha",
                cols.get("reservatorio"," "): "Reservatório/Sistema",
                cols.get("secao"," "): "Seção",
                cols.get("vazao"," "): "Vazão (L/s)",
                cols.get("obs"," "): "Observações",
            }
            st.dataframe(
                display_df[table_cols].rename(columns=renamed),
                use_container_width=True,
                height=555
            )
        else:
            st.warning("⚠️ Não encontrei as colunas necessárias para a tabela solicitada.")

    # ---------- Galeria de mídias ----------
    with col_media:
        st.markdown("**🖼️ Galeria de Mídias**")

        media_map = {
            "Foto Principal": cols.get("foto1"),
            "Foto (02)":     cols.get("foto2"),
            "Foto (03)":     cols.get("foto3"),
            "Video do Local": cols.get("video"),
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
                render_lightgallery_mixed(items, height_px=470)
            else:
                st.info("📭 Sem mídias para exibir nessa coluna. Verifique se os links estão públicos no Drive.")

    # =========================
    # MAPA — Folium (wide)
    # =========================
    st.markdown("---")
    st.subheader("🗺️ Mapa das Seções Monitoradas")

    fmap = folium.Map(location=[-5.199, -39.292], zoom_start=8, control_scale=True, prefer_canvas=True, tiles=None)

    # Bases
    folium.TileLayer("CartoDB Positron", name="🗺️ CartoDB Positron").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="🌍 OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        name="⛰️ Stamen Terrain",
        attr="Map tiles by Stamen Design (CC BY 3.0) — Data © OpenStreetMap contributors"
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="🛰️ Esri World Imagery",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
    ).add_to(fmap)

    # Grupos
    fg_bacia   = folium.FeatureGroup(name="🏞️ Bacia do Banabuiú", show=True)
    fg_trechos = folium.FeatureGroup(name="🌊 Trechos Perene", show=True)
    fg_pontos  = folium.FeatureGroup(name="📍 Pontos de Medição", show=True)

    trechos = load_geojson_safe(*TRECHOS_CAND)
    bacia   = load_geojson_safe(*BACIA_CAND)

    if trechos:
        GeoJson(
            trechos,
            name="Trechos Perene",
            style_function=lambda x: {"color": "#3498db", "weight": 4, "opacity": 0.9, "dashArray": "5, 5"},
            tooltip=GeoJsonTooltip(fields=[], aliases=[], sticky=False)
        ).add_to(fg_trechos)
        fg_trechos.add_to(fmap)

    bacia_bounds = None
    if bacia:
        GeoJson(
            bacia,
            name="Bacia do Banabuiú",
            style_function=lambda x: {"color": "#27ae60", "weight": 3, "opacity": 0.8, "fillOpacity": 0.05, "fillColor": "#27ae60"}
        ).add_to(fg_bacia)
        fg_bacia.add_to(fmap)
        bacia_bounds = geojson_bounds(bacia)

    # Pontos
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
            iframe = IFrame(html=popup_html, width=POPUP_WIDTH, height=POPUP_HEIGHT)
            popup = folium.Popup(iframe, max_width=POPUP_WIDTH)

            if USE_ICON:
                icon = CustomIcon(
                    icon_image=ICON_URL,
                    icon_size=(ICON_W, ICON_H),
                    icon_anchor=(ICON_W // 2, ICON_H)
                )
                folium.Marker(
                    location=[lat, lon],
                    icon=icon,
                    popup=popup,
                    tooltip=f"📍 {str(row.get(cols.get('secao',''), 'Seção'))}",
                ).add_to(fg_pontos)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10, color="#FF5733",
                    fill=True, fill_color="#FF5733", fill_opacity=0.9, weight=3,
                    popup=popup,
                    tooltip=f"📍 {str(row.get(cols.get('secao',''), 'Seção'))}",
                ).add_to(fg_pontos)

            pts.append((lat, lon))

        fg_pontos.add_to(fmap)

    # Fit
    if bacia_bounds:
        fmap.fit_bounds(bacia_bounds)
    elif pts:
        fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                         [max(p[0] for p in pts), max(p[1] for p in pts)]])

    LayerControl(collapsed=True, position="topright").add_to(fmap)
    st_folium(fmap, height=MAP_HEIGHT, use_container_width=True)

    # =========================
    # GRÁFICOS
    # =========================
    st.markdown("---")
    st.subheader("📈 Análises Gráficas")

    if cols.get("data") and cols.get("secao") and cols.get("vazao"):
        gdf = fdf[[cols["data"], cols["secao"], cols["vazao"]]].dropna()
        gdf[cols["vazao"]] = pd.to_numeric(gdf[cols["vazao"]].astype(str).str.replace(",", "."), errors="coerce")
        gdf = gdf.dropna(subset=[cols["vazao"]])

        # Vazão ao longo do tempo — L/s (somente datas existentes, dd/mm, ordem cronológica e pontos grandes)
        gdf[cols['data']] = pd.to_datetime(gdf[cols['data']], errors='coerce')
        gdf_plot = gdf.dropna(subset=[cols['data'], cols['vazao']]).copy()

        # Rótulo dd/mm e ordem cronológica real
        gdf_plot["data_str"] = gdf_plot[cols["data"]].dt.strftime("%d/%m")
        domain_order = (
            gdf_plot.sort_values(by=cols["data"])["data_str"]
            .drop_duplicates()
            .tolist()
        )

        st.markdown("**📈 Vazão ao Longo do Tempo**")

        # Linha
        line = (
            alt.Chart(gdf_plot)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X(
                    "data_str:O",
                    sort=domain_order,                 # ordem cronológica por rótulos
                    title="Data (dd/mm)",
                    axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida (L/s)"),
                color=alt.Color(f"{cols['secao']}:N", title="Seção", scale=alt.Scale(scheme='set1')),
                tooltip=[
                    alt.Tooltip(cols["data"], title="Data", format="%d/%m/%Y"),
                    alt.Tooltip(cols["secao"], title="Seção"),
                    alt.Tooltip(cols["vazao"], title="Vazão (L/s)", format=".2f")
                ]
            )
        )

        # Pontos destacados
        points = (
            alt.Chart(gdf_plot)
            .mark_point(size=80, filled=True)
            .encode(
                x=alt.X("data_str:O", sort=domain_order),  # mesma ordem
                y=f"{cols['vazao']}:Q",
                color=alt.Color(f"{cols['secao']}:N", title="Seção", scale=alt.Scale(scheme='set1')),
                tooltip=[
                    alt.Tooltip(cols["data"], title="Data", format="%d/%m/%Y"),
                    alt.Tooltip(cols["secao"], title="Seção"),
                    alt.Tooltip(cols["vazao"], title="Vazão (L/s)", format=".2f")
                ]
            )
        )

        chart = (line + points).properties(width="container", height=400).interactive()
        st.altair_chart(chart, use_container_width=True)

        # Boxplot — L/s
        st.markdown("**📊 Distribuição de Vazão por Seção**")
        box = alt.Chart(gdf).mark_boxplot(size=30, opacity=0.8).encode(
            x=alt.X(f"{cols['secao']}:N", title="Seção", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida (L/s)"),
            color=alt.Color(f"{cols['secao']}:N", legend=None, scale=alt.Scale(scheme='set1')),
            tooltip=[
                alt.Tooltip(cols["secao"], title="Seção"),
                alt.Tooltip(cols["vazao"], title="Vazão (L/s)", format=".2f")
            ]
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
