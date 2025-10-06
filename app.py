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
st.set_page_config(page_title="Perenização de Rios • Vazões", layout="wide")
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
    for a,b in repl.items():
        t = t.replace(a,b)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t

def first_nonempty(*vals, default=None):
    for v in vals:
        if v is not None and v != "" and not (isinstance(v, float) and math.isnan(v)):
            return v
    return default

def to_datetime_br(x):
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(x, dayfirst=dayfirst, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(x, errors="coerce")

@st.cache_data(show_spinner=False)
def load_from_gsheet_csv(sheet_id: str, gid: str = "0", sep=","):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url, sep=sep)

@st.cache_data(show_spinner=False)
def load_csv(upload, sep=","):
    return pd.read_csv(upload, sep=sep)

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

# --------- Google Drive helpers (imagens/vídeos) ----------
def gdrive_extract_id(url: str):
    if not isinstance(url, str):
        return None
    url = url.strip()
    # formatos: /file/d/<ID>/view , open?id=<ID> , uc?id=<ID> , thumbnail?id=<ID>
    m = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)
    return None

def gdrive_image_direct(url: str):
    """Transforma link do Drive em visualização direta de imagem."""
    fid = gdrive_extract_id(url)
    if fid:
        return f"https://drive.google.com/uc?export=view&id={fid}"
    return url  # pode já ser http direto

def gdrive_video_embed(url: str):
    """Transforma link do Drive em preview embed (iframe)."""
    fid = gdrive_extract_id(url)
    if fid:
        return f"https://drive.google.com/file/d/{fid}/preview"
    return None

def render_lightgallery(images: list, height_px=480):
    if not images:
        st.info("Sem imagens disponíveis.")
        return
    items_html = "\n".join(
        [f'<a class="gallery-item" href="{img["url"]}" data-sub-html="{img.get("caption","")}">'
         f'<img src="{img["thumb"]}" loading="lazy"/></a>'
         for img in images]
    )
    html = f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/css/lightgallery-bundle.min.css">
    <style>
      .lg-backdrop {{ background: rgba(0,0,0,0.88); }}
      .gallery-container {{ display:flex; flex-wrap: wrap; gap: 12px; align-items:flex-start; }}
      .gallery-item img {{ height: 140px; width: auto; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.15); }}
    </style>
    <div id="lightgallery" class="gallery-container">
      {items_html}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/lightgallery.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/plugins/zoom/lg-zoom.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightgallery@2.7.2/plugins/thumbnail/lg-thumbnail.umd.js"></script>
    <script>
      window.addEventListener('load', () => {{
        const lg = lightGallery(document.getElementById('lightgallery'), {{
          speed: 300,
          download: false,
          zoom: true,
          thumbnail: true
        }});
      }});
    </script>
    """
    components.html(html, height=height_px, scrolling=True)

def make_popup_html(row, cols):
    safe = lambda x: "-" if (x is None or (isinstance(x,float) and math.isnan(x))) else str(x)
    parts = []
    labels = {
        "campanha": "Campanha",
        "reservatorio": "Reservatório/Sistema",
        "secao": "Seção",
        "vazao": "Vazão medida",
    }
    for k in ["campanha", "reservatorio", "secao", "vazao"]:
        colname = cols.get(k)
        if colname and colname in row and pd.notna(row[colname]):
            parts.append(f"<b>{labels[k]}:</b> {safe(row[colname])}")
    return "<br>".join(parts)

def load_geojson_safe(local_path: str):
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

TRECHOS_PATH = "trechos_perene.geojson"
BACIA_PATH   = "bacia_banabuiu.geojson"

# =========================
# App
# =========================
def main():
    st.title("Monitoramento de Vazões e Perenização de Rios")
    st.caption(f"Atualizado em {datetime.now(TZ).strftime('%d/%m/%Y %H:%M:%S')} — Fuso America/Fortaleza")

    # --------- Carregamento automático do Google Sheets ---------
    with st.sidebar:
        st.header("Fonte de dados")
        st.write("Carregando automaticamente do Google Sheets.")
        sheet_id = st.text_input("Sheet ID", value="1YstNFY5ehrOUjg_AoSztcqq466uRwstKY7gpvs0BWnI")
        gid = st.text_input("GID", value="0")
        sep = st.selectbox("Separador (Sheets → CSV)", options=[",",";"], index=0)

    try:
        df = load_from_gsheet_csv(sheet_id, gid, sep=sep)
    except Exception as e:
        st.error(f"Erro ao carregar do Sheets: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("Sem dados. Verifique permissões do Sheets e o GID informado.")
        return

    df = df.replace({np.nan: None})
    cols = guess_columns(df)

    if cols.get("data") and cols["data"] in df.columns:
        df[cols["data"]] = pd.to_datetime(df[cols["data"]], errors="coerce", dayfirst=True)

    # =========================
    # FILTROS
    # =========================
    st.subheader("Filtros")

    if cols.get("data"):
        min_d = pd.to_datetime(df[cols["data"]]).min()
        max_d = pd.to_datetime(df[cols["data"]]).max()
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            data_ini = st.date_input("Data inicial", value=min_d.date() if pd.notna(min_d) else date.today())
        with c2:
            data_fim = st.date_input("Data final", value=max_d.date() if pd.notna(max_d) else date.today())
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        data_ini = data_fim = None
        st.warning("Coluna de **Data** não identificada automaticamente.", icon="⚠️")

    def options_for(colkey):
        cname = cols.get(colkey)
        if not cname or cname not in df.columns:
            return []
        vals = sorted({v for v in df[cname].dropna().tolist()})
        return vals

    with c3:
        resp = st.multiselect("Responsável", options_for("responsavel"))
    with c4:
        camp = st.multiselect("Campanha", options_for("campanha"))
    with c5:
        rese = st.multiselect("Reservatório/Sistema", options_for("reservatorio"))

    secao_opts = options_for("secao")
    sec_sel = st.multiselect("Seção", secao_opts)

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

    for key, selected in [("responsavel", resp), ("campanha", camp), ("reservatorio", rese), ("secao", sec_sel)]:
        flt = filt_in(key, selected)
        if flt is not None:
            fdf = fdf.loc[flt]

    st.success(f"Registros após filtros: **{len(fdf)}**")

    # =========================
    # TABELA + GALERIA
    # =========================
    st.subheader("Registros e Mídias")
    col_tab, col_media = st.columns([1, 1])

    with col_tab:
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
            st.warning("Não encontrei as colunas necessárias para a tabela solicitada.", icon="⚠️")

    with col_media:
        st.markdown("**Galeria** (clique para ampliar)")
        img_urls = []

        def collect_media_urls(series, caption):
            urls = []
            if series is None:
                return urls
            for v in series.dropna().unique().tolist():
                if isinstance(v, str):
                    v = v.strip()
                    if v.lower().startswith(("http://","https://")):
                        # imagens do Drive → direct view
                        direct = gdrive_image_direct(v)
                        urls.append({"url": direct, "thumb": direct, "caption": caption})
            return urls

        # fotos
        for key, label in [("foto1","Foto 1"), ("foto2","Foto 2"), ("foto3","Foto 3")]:
            cname = cols.get(key)
            if cname and cname in fdf.columns:
                img_urls.extend(collect_media_urls(fdf[cname], label))

        if img_urls:
            render_lightgallery(img_urls, height_px=420)
        else:
            st.info("Sem imagens para exibir (verifique se os links do Drive estão compartilhados como 'qualquer pessoa com o link').")

        # vídeos (Google Drive preview ou link direto/YouTube)
        vid_col = cols.get("video")
        if vid_col and vid_col in fdf.columns:
            urls = [u for u in fdf[vid_col].dropna().unique().tolist() if isinstance(u,str) and u.strip().lower().startswith(("http://","https://"))]
            if urls:
                st.markdown("**Vídeos**")
                for u in urls:
                    preview = gdrive_video_embed(u)
                    if preview:
                        components.html(f'<iframe src="{preview}" width="100%" height="340" allow="autoplay" allowfullscreen></iframe>', height=360)
                    else:
                        # YouTube ou MP4 direto
                        st.video(u)

    # =========================
    # MAPA — Folium (wide)
    # =========================
    st.subheader("Mapa das seções monitoradas")
    map_height = 520

    start_lat, start_lon, start_zoom = -5.199, -39.292, 8
    fmap = folium.Map(location=[start_lat, start_lon], zoom_start=start_zoom, control_scale=True, prefer_canvas=True)

    # Tiles com attribution correto (evita erro de custom tiles)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer("CartoDB Positron", name="CartoDB Positron").add_to(fmap)
    folium.TileLayer("CartoDB Dark_Matter", name="CartoDB Dark Matter").add_to(fmap)
    # OpenTopoMap (com attribution)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap",
        attr="Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap (CC-BY-SA)"
    ).add_to(fmap)

    # GeoJSON camadas
    trechos = load_geojson_safe(TRECHOS_PATH)
    bacia   = load_geojson_safe(BACIA_PATH)

    if trechos:
        GeoJson(
            trechos,
            name="Trechos Perene",
            style_function=lambda x: {"color":"#1f78b4","weight":3,"opacity":0.9},
            tooltip=GeoJsonTooltip(fields=[], aliases=[], sticky=False)
        ).add_to(fmap)
    else:
        st.info("Camada 'trechos_perene.geojson' não encontrada no diretório.")

    if bacia:
        GeoJson(
            bacia,
            name="Bacia do Banabuiú",
            style_function=lambda x: {"color":"#33a02c","weight":2,"opacity":0.8, "fillOpacity":0.05}
        ).add_to(fmap)
    else:
        st.info("Camada 'bacia_banabuiu.geojson' não encontrada no diretório.")

    # Pontos
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    if lat_col and lon_col and lat_col in fdf.columns and lon_col in fdf.columns:
        def to_float(v):
            if v is None: return None
            if isinstance(v, str): v = v.replace(",", ".")
            try: return float(v)
            except: return None

        pts = []
        for _, row in fdf.iterrows():
            lat = to_float(row.get(lat_col))
            lon = to_float(row.get(lon_col))
            if lat is None or lon is None:
                continue
            popup_html = make_popup_html(row, cols)
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#e31a1c",
                fill=True,
                fill_color="#fb9a99",
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=300, parse_html=True),
                tooltip=str(row.get(cols.get("secao",""), "Seção"))
            ).add_to(fmap)
            pts.append((lat,lon))

        if pts:
            fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                             [max(p[0] for p in pts), max(p[1] for p in pts)]])
    else:
        st.warning("Colunas de Latitude/Longitude não identificadas automaticamente.", icon="⚠️")

    LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=map_height, use_container_width=True)

    # =========================
    # GRÁFICOS (Data, Seção, Vazão medida)
    # =========================
    st.subheader("Gráficos")
    if cols.get("data") and cols.get("secao") and cols.get("vazao"):
        gdf = fdf[[cols["data"], cols["secao"], cols["vazao"]]].dropna()
        gdf[cols["vazao"]] = pd.to_numeric(gdf[cols["vazao"]].astype(str).str.replace(",", "."), errors="coerce")
        gdf = gdf.dropna(subset=[cols["vazao"]])

        line = alt.Chart(gdf).mark_line(point=True).encode(
            x=alt.X(f"{cols['data']}:T", title="Data"),
            y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida"),
            color=alt.Color(f"{cols['secao']}:N", title="Seção"),
            tooltip=[cols["data"], cols["secao"], cols["vazao"]]
        ).properties(width="container", height=360, title="Vazão ao longo do tempo por Seção")
        st.altair_chart(line, use_container_width=True)

        box = alt.Chart(gdf).mark_boxplot().encode(
            x=alt.X(f"{cols['secao']}:N", title="Seção"),
            y=alt.Y(f"{cols['vazao']}:Q", title="Vazão medida"),
            color=alt.Color(f"{cols['secao']}:N", legend=None)
        ).properties(width="container", height=320, title="Distribuição de Vazão por Seção")
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("Para os gráficos, são necessárias as colunas **Data**, **Seção** e **Vazão medida**.")

    st.caption("© Dados Python • Streamlit + Folium + Altair + LightGallery")

if __name__ == "__main__":
    main()
