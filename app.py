import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt, isfinite
from datetime import datetime, date
from zoneinfo import ZoneInfo
import json, re

import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG (da Secrets)
# =========================
SHEET_ID  = st.secrets.get("google_sheet_id")                  # obbligatorio
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

# Lettere di colonna (MAIUSCOLE). Default: A=Ticker, B=EPS, C=BVPS, D=Graham
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")
# Opzionali
NAME_LETTER      = st.secrets.get("name_col_letter", "")         # es. "E"
INV_URL_LETTER   = st.secrets.get("investing_col_letter", "")    # es. "F"
MORN_URL_LETTER  = st.secrets.get("morning_col_letter", "")      # es. "G"
ISIN_LETTER      = st.secrets.get("isin_col_letter", "")         # es. "H"

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup",
                   page_icon="📈", layout="wide")

# =========================
# THEME TOGGLE (Light/Dark) + Mobile tweaks
# =========================
def inject_theme_css(dark: bool):
    if dark:
        bg = "#0e1117"; paper="#161a23"; text="#e6e6e6"; sub="#bdbdbd"
        accent="#4da3ff"; border="#2a2f3a"; good="#9ad17b"; bad="#ff6b6b"; gold="#DAA520"
        metric_val = "#f2f2f2"; metric_lab = "#cfcfcf"
        formula_bg    = "#10351e"; formula_border= "#2f8f5b"; formula_text  = "#e6ffef"
    else:
        bg = "#ffffff"; paper="#fafafa"; text="#222"; sub="#666"
        accent="#0b74ff"; border="#e5e7eb"; good="#0a7f2e"; bad="#b00020"; gold="#DAA520"
        metric_val = "#111"; metric_lab = "#444"
        formula_bg    = "#e9f8ef"; formula_border= "#b8e6c9"; formula_text  = "#0d5b2a"

    st.markdown(
        f"""
        <style>
        :root {{
          --bg:{bg}; --paper:{paper}; --text:{text}; --sub:{sub};
          --accent:{accent}; --border:{border}; --good:{good}; --bad:{bad}; --gold:{gold};
          --metric-val:{metric_val}; --metric-lab:{metric_lab};
          --formula-bg:{formula_bg}; --formula-border:{formula_border}; --formula-text:{formula_text};
        }}
        .stApp {{ background-color: var(--bg); color: var(--text); }}
        h1, h2, h3, h4, h5, h6 {{ color: var(--text) !important; }}
        a, a * {{ color: var(--accent) !important; }}
        .v-card {{ background: var(--paper); border:1px solid var(--border);
                   border-radius:14px; padding:14px 16px; }}
        .v-chip-gold {{ color: var(--gold); font-weight:800; }}
        .v-sub {{ color: var(--sub); font-size:12px; }}
        [data-testid="stMetric"] > div > div:nth-child(1) {{ color: var(--metric-lab) !important; }}
        [data-testid="stMetricValue"] {{ color: var(--metric-val) !important; }}
        [data-testid="stMetricDelta"] {{ color: var(--metric-val) !important; }}
        .v-formula-title {{ font-size: 1.15rem; font-weight: 800; margin: 6px 0 8px; }}
        .v-formula-box {{
          background: var(--formula-bg);
          border: 1px solid var(--formula-border);
          border-radius: 12px; padding: 12px 14px;
          color: var(--formula-text);
        }}
        .v-formula-code {{
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
          font-size: 16px; font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

tcol1, tcol2 = st.columns([4,1], vertical_alignment="center")
with tcol1:
    st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")
with tcol2:
    dark_mode = st.toggle("🌙", value=False, help="Light/Dark mode", label_visibility="collapsed")
inject_theme_css(dark_mode)

# =========================
# GOOGLE AUTH
# =========================
@st.cache_resource
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
    else:
        # fallback locale opzionale (solo se hai committato il file; su Streamlit Cloud in genere NON serve)
        with open("service_account.json","r") as f:
            creds_dict = json.load(f)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = get_gsheet_client()
sh = gc.open_by_key(SHEET_ID)
ws_fund = sh.worksheet(FUND_TAB)
ws_hist = sh.worksheet(HIST_TAB)

# =========================
# UTILITIES
# =========================
def _letter_to_index(letter: str) -> int:
    if not letter: return -1
    s = letter.strip().upper()
    n = 0
    for ch in s:
        if not ("A" <= ch <= "Z"): return -1
        n = n*26 + (ord(ch)-64)
    return n-1

def to_number(x):
    if x is None: return None
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace("\u00A0","")
    s = s.replace("€","").replace("EUR","").replace("%","").replace("\u2212","-")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if s in {"", "-", ","}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

# === Prezzo live: TTL breve per aggiornarsi davvero ===
PRICE_TTL = int(st.secrets.get("price_ttl_seconds", 30))

@st.cache_data(show_spinner=False, ttl=PRICE_TTL)
def fetch_price_yf(symbol: str):
    try:
        t = yf.Ticker(symbol)
        # fast_info
        p = None
        fi = getattr(t, "fast_info", None)
        if fi:
            p = fi.get("last_price", None)
            if p is not None and p > 0 and isfinite(float(p)):
                return float(p)
        # 1m intraday
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
        # fallback daily
        h = t.history(period="1d")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_company_name_yf(symbol: str) -> str:
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return str(info.get("shortName") or info.get("longName") or "")
    except Exception:
        return ""

def gn_formula_225(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
    return sqrt(22.5 * eps * bvps)

DESIRED_HIST_HEADER = ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"]

def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n-1, 26)
        s = chr(65+r) + s
    return s

def ensure_history_headers():
    values = ws_hist.get_all_values()
    if not values:
        ws_hist.insert_row(DESIRED_HIST_HEADER, 1); return
    header = values[0]; changed = False
    for col in DESIRED_HIST_HEADER:
        if col not in header:
            header.append(col); changed = True
    if changed:
        ws_hist.update(f"A1:{_col_letter(len(header))}1", [header], value_input_option="USER_ENTERED")

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    ensure_history_headers()
    delta = margin = None
    if graham not in (None, "", 0) and price not in (None, ""):
        delta  = float(price) - float(graham)
        margin = (1 - (float(price)/float(graham))) * 100
    row = [
        ts, ticker,
        ("" if price is None else float(price)),
        ("" if eps is None else float(eps)),
        ("" if bvps is None else float(bvps)),
        ("" if graham is None else float(graham)),
        ("" if delta is None else float(delta)),
        ("" if margin is None else float(margin)),
        fonte
    ]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

def append_history_bulk(ts, rows):  # rows: lista di tuple (ticker, price, eps, bvps, graham, fonte)
    ensure_history_headers()
    out = []
    for ticker, price, eps, bvps, graham, fonte in rows:
        delta = margin = None
        if graham not in (None, "", 0) and price not in (None, ""):
            delta  = float(price) - float(graham)
            margin = (1 - (float(price)/float(graham))) * 100
        out.append([
            ts, ticker,
            ("" if price is None else float(price)),
            ("" if eps is None else float(eps)),
            ("" if bvps is None else float(bvps)),
            ("" if graham is None else float(graham)),
            ("" if delta is None else float(delta)),
            ("" if margin is None else float(margin)),
            fonte
        ])
    if out:
        ws_hist.append_rows(out, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def load_history():
    ensure_history_headers()
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty: return dfh
    if "Timestamp" in dfh.columns:
        dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"], errors="coerce")
    for col in ["Delta","MarginPct"]:
        if col not in dfh.columns: dfh[col] = np.nan
    if not dfh.empty:
        _p = pd.to_numeric(dfh.get("Price"), errors="coerce")
        _g = pd.to_numeric(dfh.get("Graham"), errors="coerce")
        mask = dfh["Delta"].isna() | dfh["MarginPct"].isna()
        dfh.loc[mask, "Delta"] = (_p - _g).where((_p.notna()) & (_g.notna()))
        dfh.loc[mask, "MarginPct"] = (1 - (_p/_g))*100
    return dfh

# =========================
# LOAD FUNDAMENTALS (per lettera)
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(), {}
    header, data = values[0], values[1:]

    def idx(letter):
        return _letter_to_index(letter) if letter else -1

    idx_ticker = idx(TICKER_LETTER)
    idx_eps    = idx(EPS_LETTER)
    idx_bvps   = idx(BVPS_LETTER)
    idx_gn     = idx(GN_LETTER)
    idx_name   = idx(NAME_LETTER)
    idx_inv    = idx(INV_URL_LETTER)
    idx_morn   = idx(MORN_URL_LETTER)
    idx_isin   = idx(ISIN_LETTER)

    df = pd.DataFrame({
        "Ticker_raw":  [row[idx_ticker] if idx_ticker >= 0 and idx_ticker < len(row) else "" for row in data],
        "EPS_raw":     [row[idx_eps]    if idx_eps    >= 0 and idx_eps    < len(row) else "" for row in data],
        "BVPS_raw":    [row[idx_bvps]   if idx_bvps   >= 0 and idx_bvps   < len(row) else "" for row in data],
        "GN_sheet_raw":[row[idx_gn]     if idx_gn     >= 0 and idx_gn     < len(row) else "" for row in data],
    })
    df["Name_raw"]        = [row[idx_name] if (idx_name >= 0 and idx_name < len(row)) else "" for row in data] if idx_name >= 0 else ""
    df["InvestingURL_raw"]= [row[idx_inv]  if (idx_inv  >=0 and idx_inv  < len(row)) else "" for row in data] if idx_inv  >= 0 else ""
    df["MorningURL_raw"]  = [row[idx_morn] if (idx_morn >=0 and idx_morn < len(row)) else "" for row in data] if idx_morn >= 0 else ""
    df["ISIN_raw"]        = [row[idx_isin] if (idx_isin >=0 and idx_isin < len(row)) else "" for row in data] if idx_isin >= 0 else ""

    df = df[(df["Ticker_raw"].astype(str).str.strip()!="")].reset_index(drop=True)
    df["Ticker"]    = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["EPS"]       = df["EPS_raw"].apply(to_number)
    df["BVPS"]      = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"]  = df["GN_sheet_raw"].apply(to_number)
    df["Name"]      = (df["Name_raw"].astype(str).str.strip() if isinstance(df["Name_raw"], pd.Series) else "")
    df["InvestingURL"] = (df["InvestingURL_raw"].astype(str).str.strip() if isinstance(df["InvestingURL_raw"], pd.Series) else "")
    df["MorningURL"]   = (df["MorningURL_raw"].astype(str).str.strip() if isinstance(df["MorningURL_raw"], pd.Series) else "")
    df["ISIN"]         = (df["ISIN_raw"].astype(str).str.strip() if isinstance(df["ISIN_raw"], pd.Series) else "")

    return df, {"has_name": idx_name>=0, "has_inv": idx_inv>=0, "has_morn": idx_morn>=0, "has_isin": idx_isin>=0}

# =========================
# UI
# =========================
df, meta = load_fundamentals_by_letter()
if df.empty:
    st.warning("Nessun dato utile. Controlla il foglio.")
else:
    @st.cache_data(show_spinner=False, ttl=86400)
    def get_display_name(tick: str) -> str:
        r = df[df["Ticker"] == tick]
        if not r.empty:
            n = str(r.iloc[0].get("Name") or "").strip()
            if n: return n
        return fetch_company_name_yf(normalize_symbol(tick)) or ""

    tickers_all = sorted(df["Ticker"].tolist())
    labels = []
    label_to_ticker = {}
    for t in tickers_all:
        nm = get_display_name(t)
        lab = f"{t} — {nm}" if nm else t
        labels.append(lab); label_to_ticker[lab] = t

    tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

    with tab1:
        selected_label = st.selectbox("Scegli il Ticker", options=labels, index=0)
        tick = label_to_ticker[selected_label]

        row = df[df["Ticker"]==tick].iloc[0]
        eps_val, bvps_val, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_applied = gn_formula_225(eps_val, bvps_val)
        symbol = normalize_symbol(tick)
        price_live = fetch_price_yf(symbol)

        company_name = get_display_name(tick)
        isin = str(row.get("ISIN") or "").strip()

        # Link
        yahoo_url = f"https://finance.yahoo.com/quote/{tick}"
        inv_url   = str(row.get("InvestingURL") or "").strip()
        mor_url   = str(row.get("MorningURL") or "").strip()
        query_key = isin if isin else tick
        if not inv_url:
            inv_url = f"https://it.investing.com/search/?q={query_key}"
        if not mor_url:
            mor_url = f"https://www.morningstar.com/search?query={query_key}"

        # Header
        st.markdown(
            f"""
            <div class="v-card" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
              <div style="display:flex;flex-direction:column;gap:2px;">
                <h3 style="margin:0">{tick} — {company_name}</h3>
                {"<div class='v-sub'>ISIN: "+isin+"</div>" if isin else ""}
              </div>
              <span class="v-links" style="display:inline-flex;gap:10px;align-items:center; margin-left:4px;">
                <a class="yf"  href="{yahoo_url}" target="_blank" rel="noopener" title="Yahoo Finance">
                  <img src="https://www.google.com/s2/favicons?sz=64&domain=finance.yahoo.com">
                </a>
                <a class="dim" href="{inv_url}" target="_blank" rel="noopener" title="Investing">
                  <img src="https://www.google.com/s2/favicons?sz=64&domain=it.investing.com">
                </a>
                <a class="dim" href="{mor_url}" target="_blank" rel="noopener" title="Morningstar">
                  <img src="https://www.google.com/s2/favicons?sz=64&domain=morningstar.com">
                </a>
              </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Controlli refresh
        r1, r2 = st.columns([1,3])
        with r1:
            if st.button("🔄 Aggiorna ora"):
                st.cache_data.clear()
                st.rerun()
        with r2:
            st.caption(f"Aggiornamento prezzo con TTL **{PRICE_TTL}s** (evita rate-limit Yahoo).")

        # Metriche
        margin_pct = (1 - (price_live/gn_sheet))*100 if (price_live is not None and gn_sheet not in (None,0)) else None
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
        with c2: st.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
        with c3:
            if margin_pct is not None:
                pct = f"{margin_pct:.2f}%"
                html = (
                    f"<div style='text-align:center;'><div style='font-weight:800; font-size:20px; color:var(--good);'>{pct}</div>"
                    f"<div style='margin-top:4px; font-weight:700; color:{'var(--good)' if margin_pct>0 else 'var(--bad)'};'>"
                    f"{'Sottovalutata' if margin_pct>0 else 'Sopravvalutata'}</div></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.metric("Margine", "n/d")

        # Formula applicata
        st.markdown('<div class="v-formula-title">The GN Formula (Applied)</div>', unsafe_allow_html=True)
        if gn_applied is not None:
            st.markdown(
                f"""
                <div class="v-formula-box">
                  <div class="v-formula-code">
                    √(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_applied:.4f}
                  </div>
                  <div class="v-sub">EPS e BVPS dal foglio; coefficiente 22.5 (Graham)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        st.markdown("---")
        is_admin = st.toggle("🛠️ Modalità amministratore", value=True,
                             help="Mostra/nasconde i comandi di amministrazione")

        if is_admin:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🔄 Aggiorna dal foglio", use_container_width=True):
                    st.cache_data.clear(); st.rerun()
            with col2:
                if st.button("✍️ Riscrivi Graham# su Sheet", use_container_width=True):
                    gn_series = df.apply(lambda r: gn_formula_225(r["EPS"], r["BVPS"]), axis=1)
                    out = [[("" if (v is None or pd.isna(v)) else float(v))] for v in gn_series]
                    start_row = 2
                    end_row = start_row + len(out) - 1
                    ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}", out,
                                   value_input_option="USER_ENTERED")
                    st.success("Colonna Graham# aggiornata (22,5×EPS×BVPS).")
                    st.cache_data.clear(); st.rerun()
            with col3:
                if st.button("💾 Salva snapshot", use_container_width=True):
                    now_str = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M:%S")
                    append_history_row(now_str, tick, price_live, eps_val, bvps_val, gn_sheet, "App (GN da Sheet)")
                    st.success("Snapshot salvato nello Storico.")
            with col4:
                if st.button("🗂️ Snapshot TUTTI i titoli", use_container_width=True):
                    now_str = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M:%S")
                    rows_out = []
                    for _, r in df.iterrows():
                        tck = str(r["Ticker"]).strip().upper()
                        if not tck: continue
                        px = fetch_price_yf(normalize_symbol(tck))
                        eps_i, bvps_i, gn_i = r["EPS"], r["BVPS"], r["GN_sheet"]
                        rows_out.append((tck, px, eps_i, bvps_i, gn_i, "App (bulk)"))
                    append_history_bulk(now_str, rows_out)
                    st.success(f"Snapshot di {len(rows_out)} titoli salvato ✅")

    with tab2:
        dfh = load_history()
        if not dfh.empty:
            try:
                current_tick = label_to_ticker[selected_label]
            except Exception:
                current_tick = None

            dft = dfh[dfh["Ticker"].astype(str).str.upper() == (current_tick or "").upper()] if current_tick else dfh
            dft = dft.sort_values("Timestamp")

            if not dft.empty and pd.notna(dft.iloc[-1].get("Timestamp")):
                st.success(f"✅ Ultimo snapshot: {pd.to_datetime(dft.iloc[-1]['Timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")

            if not dft.empty:
                min_day = pd.to_datetime(dft["Timestamp"]).dt.date.min()
                max_day = pd.to_datetime(dft["Timestamp"]).dt.date.max()
                start, end = st.date_input("Intervallo date",
                                           value=(min_day, max_day),
                                           min_value=min_day, max_value=max_day)
                if isinstance(start, date) and isinstance(end, date) and start <= end:
                    dft = dft[(pd.to_datetime(dft["Timestamp"]).dt.date >= start) &
                              (pd.to_datetime(dft["Timestamp"]).dt.date <= end)]

                show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"] if c in dft.columns]
                st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

                csv = dft[show_cols].to_csv(index=False).encode("utf-8")
                fname = f"storico_{(current_tick or 'ALL')}.csv"
                st.download_button("⬇️ Scarica CSV", data=csv, file_name=fname, mime="text/csv")

                if ("Price" in dft.columns) and ("Graham" in dft.columns):
                    plot_df = dft[["Timestamp","Price","Graham"]].dropna().set_index("Timestamp")
                    st.line_chart(plot_df, use_container_width=True)
        else:
            st.info("La tab Storico è vuota.")

