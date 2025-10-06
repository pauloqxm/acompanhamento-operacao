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
      .gallery-item img {{ height: 140px; width: auto; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.15); }}
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
    parts = []
    for k in ["campanha", "reservatorio", "secao", "vazao"]:
        colname = cols.get(k)
        if colname and colname in row and pd.notna(row[colname]):
            parts.append(f"<b>{labels[k]}:</b> {safe(row[colname])}")
    return "<br>".join(parts)

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
    st.title("Monitoramento de Vazões e Perenização de Rios")
    st.caption(f"Atualizado em {datetime.now(TZ).strftime('%d/%m/%Y %H:%M:%S')} — Fuso America/Fortaleza")

    # Carregamento automático do Google Sheets
    with st.sidebar:
        st.header("Fonte de dados")
        st.write("Carregando automaticamente do Google Sheets.")
        sheet_id = st.text_input("Sheet ID", value="1YstNFY5ehrOUjg_AoSztcqq466uRwstKY7gpvs0BWnI")
        gid = st.text_input("GID", value="0")
        sep = st.selectbox("Separador (Sheets → CSV)", options=[",",";"], index=0)

        st.divider()
        st.subheader("Camadas de mapa (arquivos locais)")
        st.write(f"trechos_perene.geojson: {'✅' if load_geojson_safe(*TRECHOS_CAND) else '❌ não encontrado'}")
        st.write(f"bacia_banabuiu.geojson: {'✅' if load_geojson_safe(*BACIA_CAND) else '❌ não encontrado'}")

    try:
        df = load_from_gsheet_csv(sheet_id, gid, sep=sep)
    except Exception as e:
        st.error(f"Erro ao carregar do Sheets: {e}")
        return

    if df.empty:
        st.info("Sem dados. Verifique permissões do Sheets e o GID informado.")
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
    # TABELA + MÍDIAS (seletor)
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
        st.markdown("**Mídias**")

        media_map = {
            "Foto do local_URL": cols.get("foto1"),
            "Foto (02)_URL":    cols.get("foto2"),
            "Foto (03)_URL":    cols.get("foto3"),
            "Video do Local_URL": cols.get("video"),
        }
        valid_options = [label for label, cname in media_map.items() if cname and cname in fdf.columns]
        if not valid_options:
            st.info("Nenhuma coluna de mídia encontrada na base.")
        else:
            choice = st.selectbox("Escolha o que exibir", valid_options, index=0)

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
                st.info("Sem mídias para exibir nessa coluna. Verifique se os links apontam para arquivos do Drive (não pastas) e se estão compartilhados como 'qualquer pessoa com o link'.")

    # =========================
    # MAPA — Folium (wide)
    # =========================
    st.subheader("Mapa das seções monitoradas")
    map_height = 520

    # Mapa base (sem fit ainda)
    fmap = folium.Map(location=[-5.199, -39.292], zoom_start=8, control_scale=True, prefer_canvas=True)

    # Tiles (bases) com attribution
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer("CartoDB Positron", name="CartoDB Positron").add_to(fmap)
    folium.TileLayer("CartoDB Dark_Matter", name="CartoDB Dark Matter").add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap",
        attr="Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap (CC-BY-SA)"
    ).add_to(fmap)

    # FeatureGroups para poder ligar/desligar
    fg_bacia   = folium.FeatureGroup(name="Bacia do Banabuiú", show=True)
    fg_trechos = folium.FeatureGroup(name="Trechos Perene", show=True)
    fg_pontos  = folium.FeatureGroup(name="Pontos de Medição", show=True)

    # GeoJSON camadas
    trechos = load_geojson_safe(*TRECHOS_CAND)
    bacia   = load_geojson_safe(*BACIA_CAND)

    if trechos:
        GeoJson(
            trechos,
            name="Trechos Perene",
            style_function=lambda x: {"color":"#1f78b4","weight":3,"opacity":0.9},
            tooltip=GeoJsonTooltip(fields=[], aliases=[], sticky=False)
        ).add_to(fg_trechos)
        fg_trechos.add_to(fmap)
    else:
        st.info("Camada 'trechos_perene.geojson' não encontrada.")

    bacia_bounds = None
    if bacia:
        GeoJson(
            bacia,
            name="Bacia do Banabuiú",
            style_function=lambda x: {"color":"#33a02c","weight":2,"opacity":0.8, "fillOpacity":0.05}
        ).add_to(fg_bacia)
        fg_bacia.add_to(fmap)
        bacia_bounds = geojson_bounds(bacia)
    else:
        st.info("Camada 'bacia_banabuiu.geojson' não encontrada.")

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
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#e31a1c",
                fill=True,
                fill_color="#fb9a99",
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=300, parse_html=True),
                tooltip=str(row.get(cols.get("secao",""), "Seção"))
            ).add_to(fg_pontos)
            pts.append((lat, lon))
        fg_pontos.add_to(fmap)
    else:
        st.warning("Colunas de Latitude/Longitude não identificadas automaticamente.", icon="⚠️")

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
