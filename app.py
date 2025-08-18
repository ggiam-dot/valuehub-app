import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt, isfinite
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo
import json, re

import gspread
from google.oauth2.service_account import Credentials
from streamlit_autorefresh import st_autorefresh   # NEW

# =========================
# CONFIG (da Secrets)
# =========================
SHEET_ID  = st.secrets.get("google_sheet_id")                  # obbligatorio
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")
MIB_SYMBOL = st.secrets.get("mib_symbol", "^FTSEMIB")
BORSA_LINK = st.secrets.get("borsa_link", "https://www.borsaitaliana.it/borsa/indice/ftse-mib/dettaglio.html")

# Lettere di colonna
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")
NAME_LETTER   = st.secrets.get("name_col_letter", "")
INV_URL_LETTER= st.secrets.get("investing_col_letter", "")
MORN_URL_LETTER=st.secrets.get("morning_col_letter", "")
ISIN_LETTER   = st.secrets.get("isin_col_letter", "")

# Prezzi: TTL breve per evitare rate-limit; puoi cambiare nei secrets
PRICE_TTL = int(st.secrets.get("price_ttl_seconds", 30))

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup",
                   page_icon="📈", layout="wide")

# =========================
# THEME & CSS (mobile-first + pulsanti armonici)
# =========================
if "dark" not in st.session_state:
    st.session_state.dark = False

def inject_theme_css(dark: bool):
    if dark:
        bg="#0e1117"; paper="#161a23"; text="#e6e6e6"; sub="#bdbdbd"
        accent="#4da3ff"; border="#2a2f3a"; good="#9ad17b"; bad="#ff6b6b"; gold="#DAA520"
        metric_val="#f2f2f2"; metric_lab="#cfcfcf"
        formula_bg="#10351e"; formula_border="#2f8f5b"; formula_text="#e6ffef"; stripe_bg="#121723"
        pill_bg="#1e2635"; btn_bg="#1b2332"; btn_txt="#e6e6e6"
    else:
        bg="#ffffff"; paper="#fafafa"; text="#222"; sub="#666"
        accent="#0b74ff"; border="#e5e7eb"; good="#0a7f2e"; bad="#b00020"; gold="#DAA520"
        metric_val="#111"; metric_lab="#444"
        formula_bg="#e9f8ef"; formula_border="#b8e6c9"; formula_text="#0d5b2a"; stripe_bg="#f7f7f8"
        pill_bg="#eef2ff"; btn_bg="#ffffff"; btn_txt="#111"

    st.markdown(f"""
    <style>
    :root {{
      --bg:{bg}; --paper:{paper}; --text:{text}; --sub:{sub};
      --accent:{accent}; --border:{border}; --good:{good}; --bad:{bad}; --gold:{gold};
      --metric-val:{metric_val}; --metric-lab:{metric_lab};
      --formula-bg:{formula_bg}; --formula-border:{formula_border}; --formula-text:{formula_text};
      --stripe-bg:{stripe_bg}; --pill-bg:{pill_bg};
      --btn-bg:{btn_bg}; --btn-txt:{btn_txt};
    }}
    .stApp {{ background-color: var(--bg); color: var(--text); }}

    /* Titolo snello */
    .v-title {{ font-weight: 800; font-size: 1.35rem; line-height: 1.2; margin: 0 0 4px 0; }}
    .v-title .v-title-light {{ font-weight: 600; opacity: .9; }}
    @media (max-width: 640px) {{ .v-title {{ font-size: 1.15rem; }} }}

    .v-card {{ background: var(--paper); border:1px solid var(--border);
               border-radius:14px; padding:14px 16px; }}
    .v-sub {{ color: var(--sub); font-size:12px; }}

    .v-links {{ display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
    .v-link {{ display:flex; gap:8px; align-items:center; font-size:14px; }}
    .v-link img {{ width:20px; height:20px; }}

    .pill {{ background:var(--pill-bg); padding:4px 10px; border-radius:999px;
             border:1px solid var(--border); font-weight:600; }}

    .btn-link {{ background:var(--btn-bg); color:var(--btn-txt) !important; border:1px solid var(--border);
                 padding:8px 12px; border-radius:10px; text-decoration:none; display:inline-block; }}
    .btn-link:hover {{ border-color:var(--accent); }}

    .stripe {{
      background: var(--stripe-bg); border: 1px solid var(--border); border-radius: 12px;
      padding: 8px 12px; display:flex; align-items:center; justify-content:space-between; gap: 10px;
      margin-bottom: 10px; flex-wrap: wrap;
    }}
    .pct-pos {{ color: var(--good); font-weight:700; }}
    .pct-neg {{ color: var(--bad);  font-weight:700; }}

    .v-formula-title {{ font-size: 1.05rem; font-weight:800; margin: 6px 0 8px; }}
    .v-formula-box {{ background: var(--formula-bg); border:1px solid var(--formula-border);
                      border-radius:12px; padding: 12px 14px; color: var(--formula-text); }}
    .v-formula-code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:15px; font-weight:700; }}

    /* Pulsanti Streamlit uniformi */
    .stButton>button {{
      width: 100%;
      background: var(--btn-bg); color: var(--btn-txt);
      border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 14px; font-weight: 600;
    }}
    .stButton>button:hover {{ border-color: var(--accent); }}

    /* Mobile tweaks */
    @media (max-width: 640px) {{
      .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    }}
    </style>
    """, unsafe_allow_html=True)

header_l, header_r = st.columns([4,1], vertical_alignment="center")
with header_l:
    st.markdown(
        "<div class='v-title'>📈 Vigil – Value Investment Graham <span class='v-title-light'>(Intelligent)</span> Lookup</div>",
        unsafe_allow_html=True
    )
with header_r:
    st.session_state.dark = st.toggle("🌙", value=st.session_state.dark, help="Light/Dark mode", label_visibility="collapsed")
inject_theme_css(st.session_state.dark)

# =========================
# MARKET TIME / GOOGLE AUTH
# =========================
ROME = ZoneInfo("Europe/Rome")
def is_it_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(ROME)
    if now.weekday() >= 5:  # sab/dom
        return False
    return dtime(9,0) <= now.time() <= dtime(17,35)

@st.cache_resource
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
    else:
        with open("service_account.json","r") as f:
            creds_dict = json.load(f)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = get_gsheet_client()
sh = gc.open_by_key(SHEET_ID)
ws_fund = sh.worksheet(FUND_TAB)
ws_hist = sh.worksheet(HIST_TAB)

# =========================
# UTILS
# =========================
ISIN_REGEX = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")

def _letter_to_index(letter: str) -> int:
    if not letter: return -1
    s = letter.strip().upper(); n = 0
    for ch in s:
        if not ("A" <= ch <= "Z"): return -1
        n = n*26 + (ord(ch)-64)
    return n-1

def to_number(x):
    if x is None: return None
    if isinstance(x,(int,float)): return float(x)
    s = str(x).strip().replace("\u00A0","")
    s = s.replace("€","").replace("EUR","").replace("%","").replace("\u2212","-")
    s = re.sub(r"[^0-9\-,\.]","",s)
    if s in {"","-","." ,","}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): s = s.replace(".","").replace(",",".")
        else: s = s.replace(",","")
    elif "," in s: s = s.replace(",",".")
    try: return float(s)
    except: return None

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

# --- formattazioni (3 decimali per i prezzi) ---
def fmt_it(x, dec=2):
    if x is None or (isinstance(x, float) and (np.isnan(x))): return ""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- Prezzi ---
@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def price_live(symbol: str):
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t,"fast_info",None)
        if fi:
            p = fi.get("last_price")
            if p and p>0 and isfinite(float(p)): return float(p)
        h = t.history(period="1d", interval="1m")
        if not h.empty: return float(h["Close"].dropna().iloc[-1])
        h = t.history(period="1d")
        if not h.empty: return float(h["Close"].dropna().iloc[-1])
    except: pass
    return None

@st.cache_data(ttl=300, show_spinner=False)
def price_close(symbol: str):
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="5d", interval="1d")
        if not h.empty: return float(h["Close"].dropna().iloc[-1])
    except: pass
    return None

def get_price(symbol: str, mode: str):  # mode: live | close | auto
    if mode == "live":  return price_live(symbol)
    if mode == "close": return price_close(symbol)
    return price_live(symbol) if is_it_market_open() else price_close(symbol)

# FTSE MIB
@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def mib_quote(mode: str):
    t = yf.Ticker(MIB_SYMBOL)
    last = prev = None
    try:
        if mode=="live":
            fi = getattr(t,"fast_info",None)
            if fi:
                last = fi.get("last_price")
                prev = fi.get("previous_close") or fi.get("regular_market_previous_close")
        if last is None or prev is None:
            h = t.history(period="2d", interval="1d")
            if not h.empty:
                last = float(h["Close"].dropna().iloc[-1])
                if len(h)>=2: prev = float(h["Close"].dropna().iloc[-2])
    except: pass
    if last is None: return None, None
    pct = None if not prev else ((float(last)-float(prev))/float(prev))*100
    return float(last), (None if pct is None else float(pct))

@st.cache_data(ttl=86400, show_spinner=False)
def company_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info
        return str(info.get("shortName") or info.get("longName") or "")
    except: return ""

def gn_225(eps,bvps):
    if eps and bvps and eps>0 and bvps>0: return sqrt(22.5*eps*bvps)
    return None

DESIRED_HIST_HEADER = ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"]

def ensure_history_headers():
    values = ws_hist.get_all_values()
    if not values:
        ws_hist.insert_row(DESIRED_HIST_HEADER,1); return
    header = values[0]; changed=False
    for col in DESIRED_HIST_HEADER:
        if col not in header: header.append(col); changed=True
    if changed: ws_hist.update("A1:I1", [header], value_input_option="USER_ENTERED")

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    ensure_history_headers()
    delta = margin = None
    if graham not in (None,"",0) and price not in (None,""):
        delta = float(price) - float(graham)
        margin = (1 - (float(price)/float(graham)))*100
    row = [ts, ticker,
           ("" if price is None else float(price)),
           ("" if eps is None else float(eps)),
           ("" if bvps is None else float(bvps)),
           ("" if graham is None else float(graham)),
           ("" if delta is None else float(delta)),
           ("" if margin is None else float(margin)),
           fonte]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

def append_history_bulk(ts, rows):
    ensure_history_headers()
    out=[]
    for ticker, price, eps, bvps, graham, fonte in rows:
        delta=margin=None
        if graham not in (None,"",0) and price not in (None,""):
            delta = float(price) - float(graham)
            margin = (1 - (float(price)/float(graham)))*100
        out.append([ts, ticker,
                    ("" if price is None else float(price)),
                    ("" if eps is None else float(eps)),
                    ("" if bvps is None else float(bvps)),
                    ("" if graham is None else float(graham)),
                    ("" if delta is None else float(delta)),
                    ("" if margin is None else float(margin)),
                    fonte])
    if out: ws_hist.append_rows(out, value_input_option="USER_ENTERED")

# ============ LOAD FUNDAMENTALS ============
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values)<2: return pd.DataFrame(), {}
    _, data = values[0], values[1:]
    def idx(letter): return _letter_to_index(letter) if letter else -1
    idx_t, idx_e, idx_b, idx_g = idx(TICKER_LETTER), idx(EPS_LETTER), idx(BVPS_LETTER), idx(GN_LETTER)
    idx_n, idx_i, idx_m, idx_is = idx(NAME_LETTER), idx(INV_URL_LETTER), idx(MORN_URL_LETTER), idx(ISIN_LETTER)
    df = pd.DataFrame({
        "Ticker_raw":[row[idx_t] if 0<=idx_t<len(row) else "" for row in data],
        "EPS_raw":[row[idx_e] if 0<=idx_e<len(row) else "" for row in data],
        "BVPS_raw":[row[idx_b] if 0<=idx_b<len(row) else "" for row in data],
        "GN_sheet_raw":[row[idx_g] if 0<=idx_g<len(row) else "" for row in data],
    })
    df["Name_raw"]        = [row[idx_n]  if 0<=idx_n <len(row) else "" for row in data] if idx_n  >=0 else ""
    df["InvestingURL_raw"]= [row[idx_i]  if 0<=idx_i <len(row) else "" for row in data] if idx_i  >=0 else ""
    df["MorningURL_raw"]  = [row[idx_m]  if 0<=idx_m <len(row) else "" for row in data] if idx_m  >=0 else ""
    df["ISIN_raw"]        = [row[idx_is] if 0<=idx_is<len(row) else "" for row in data] if idx_is>=0 else ""

    df = df[(df["Ticker_raw"].astype(str).str.strip()!="")].reset_index(drop=True)
    df["Ticker"]   = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["EPS"]      = df["EPS_raw"].apply(to_number)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number)
    df["Name"]     = (df["Name_raw"].astype(str).str.strip() if isinstance(df["Name_raw"], pd.Series) else "")
    df["InvestingURL"] = (df["InvestingURL_raw"].astype(str).str.strip() if isinstance(df["InvestingURL_raw"], pd.Series) else "")
    df["MorningURL"]   = (df["MorningURL_raw"].astype(str).str.strip() if isinstance(df["MorningURL_raw"], pd.Series) else "")
    df["ISIN"]         = (df["ISIN_raw"].astype(str).str.strip() if isinstance(df["ISIN_raw"], pd.Series) else "")

    return df, {}

# ============ UI ============
df, _ = load_fundamentals_by_letter()

# --- FTSE MIB STRIPE ---
mib_mode_default = "live" if is_it_market_open() else "close"
mib_last, mib_pct = mib_quote(mib_mode_default)
status = "Aperto" if mib_mode_default=="live" else "Chiuso"
pct_html = "" if mib_pct is None else f"<span class='{'pct-pos' if mib_pct>=0 else 'pct-neg'}'>{mib_pct:+.2f}%</span>"
st.markdown(f"""
<div class="stripe">
  <div class="pill">🇮🇹 FTSE MIB · {status}</div>
  <div style="font-weight:700">{ (fmt_it(mib_last,3) if mib_last is not None else "n/d") } {pct_html}</div>
  <div><a class="btn-link" href="{BORSA_LINK}" target="_blank" rel="noopener">Borsa Italiana ↗︎</a></div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("Nessun dato utile. Controlla il foglio.")
else:
    @st.cache_data(show_spinner=False, ttl=86400)
    def get_display_name(t: str) -> str:
        r = df[df["Ticker"]==t]
        if not r.empty:
            n = str(r.iloc[0].get("Name") or "").strip()
            if n: return n
        try:
            info = yf.Ticker(normalize_symbol(t)).info
            return str(info.get("shortName") or info.get("longName") or "")
        except: return ""

    tickers_all = sorted(df["Ticker"].tolist())
    label_to_ticker = {}
    labels=[]
    for t in tickers_all:
        nm = get_display_name(t)
        lab = f"{t} — {nm}" if nm else t
        labels.append(lab); label_to_ticker[lab]=t

    tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

    with tab1:
        selected_label = st.selectbox("Scegli il Ticker", options=labels, index=0)
        tick = label_to_ticker[selected_label]

        row = df[df["Ticker"]==tick].iloc[0]
        eps_val, bvps_val, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_applied = gn_225(eps_val, bvps_val)
        symbol = normalize_symbol(tick)

        # --- controlli refresh / origine prezzo
        c_ref1, c_ref2, c_ref3 = st.columns([1.2,1,2])
        with c_ref1:
            price_mode = st.radio("Origine prezzo", ["Auto","Intraday","Chiusura"], horizontal=True, index=0,
                                  help="Auto: intraday in orario di mercato, altrimenti chiusura precedente")
        with c_ref2:
            if st.button("🔄 Aggiorna ora"): st.cache_data.clear(); st.rerun()
        with c_ref3:
            auto = st.toggle("Auto-refresh 60s", value=False, help="Aggiorna automaticamente i prezzi")
            if auto:
                st_autorefresh(interval=60_000, key="auto-refresh")

        mode = {"Auto":"auto","Intraday":"live","Chiusura":"close"}[price_mode]
        price_val = get_price(symbol, mode)
        mode_badge = {"auto":"Auto","live":"Intraday","close":"Chiusura"}[mode]

        company = get_display_name(tick)

        # --- Link con ISIN validato
        isin_raw = str(row.get("ISIN") or "").strip().upper()
        isin = isin_raw if ISIN_REGEX.match(isin_raw) else ""
        query_key = isin if isin else tick

        yahoo_url = f"https://finance.yahoo.com/quote/{tick}"
        inv_url   = (row.get("InvestingURL") or "").strip() or f"https://it.investing.com/search/?q={query_key}"
        mor_url   = (row.get("MorningURL")  or "").strip() or f"https://www.morningstar.com/search?query={query_key}"

        st.markdown(f"""
        <div class="v-card" style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
          <div>
            <h3 style="margin:0">{tick} — {company}</h3>
            {"<div class='v-sub'>ISIN: "+isin+"</div>" if isin else ""}
          </div>
          <div class="v-links">
            <a class="v-link" href="{yahoo_url}" target="_blank" rel="noopener">
              <img src="https://www.google.com/s2/favicons?sz=64&domain=finance.yahoo.com"><span>Yahoo</span>
            </a>
            <a class="v-link" href="{inv_url}" target="_blank" rel="noopener">
              <img src="https://www.google.com/s2/favicons?sz=64&domain=it.investing.com"><span>Investing</span>
            </a>
            <a class="v-link" href="{mor_url}" target="_blank" rel="noopener">
              <img src="https://www.google.com/s2/favicons?sz=64&domain=morningstar.com"><span>Morningstar</span>
            </a>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Metriche (prezzi 3 decimali)
        margin_pct = (1 - (price_val/gn_sheet))*100 if (price_val is not None and gn_sheet not in (None,0)) else None
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric(f"Prezzo ({mode_badge})", fmt_it(price_val,3) if price_val is not None else "n/d")
        with c2: st.metric("Graham#", fmt_it(gn_sheet,2) if gn_sheet is not None else "n/d")
        with c3:
            if margin_pct is not None:
                st.metric("Margine", fmt_it(margin_pct,2) + "%")
            else:
                st.metric("Margine","n/d")
        with c4:
            st.metric("GN (da EPS×BVPS)", fmt_it(gn_applied,2) if gn_applied is not None else "n/d")

        # --- Formula
        st.markdown('<div class="v-formula-title">The GN Formula (Applied)</div>', unsafe_allow_html=True)
        if gn_applied is not None:
            st.markdown(
                f"<div class='v-formula-box'><div class='v-formula-code'>√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_applied:.4f}</div><div class='v-sub'>EPS e BVPS dal foglio; coefficiente 22.5 (Graham)</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        st.markdown("---")
        is_admin = st.toggle("🛠️ Modalità amministratore", value=True, help="Comandi di scrittura sul foglio")
        if is_admin:
            col1,col2,col3,col4 = st.columns(4)
            if col1.button("🔄 Aggiorna dal foglio", use_container_width=True):
                st.cache_data.clear(); st.rerun()
            if col2.button("✍️ Riscrivi Graham# su Sheet", use_container_width=True):
                gn_series = df.apply(lambda r: gn_225(r["EPS"], r["BVPS"]), axis=1)
                out = [[("" if (v is None or pd.isna(v)) else float(v))] for v in gn_series]
                start_row = 2; end_row = start_row + len(out) - 1
                ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}", out, value_input_option="USER_ENTERED")
                st.success("Colonna Graham# aggiornata (22.5×EPS×BVPS)."); st.cache_data.clear(); st.rerun()
            if col3.button("💾 Salva snapshot", use_container_width=True):
                now_str = datetime.now(ROME).strftime("%Y-%m-%d %H:%M:%S")
                append_history_row(now_str, tick, price_val, eps_val, bvps_val, gn_sheet, f"App ({mode_badge})")
                st.success("Snapshot salvato.")
            if col4.button("🗂️ Snapshot TUTTI i titoli", use_container_width=True):
                now_str = datetime.now(ROME).strftime("%Y-%m-%d %H:%M:%S")
                rows_out=[]
                for _,r in df.iterrows():
                    tck = str(r["Ticker"]).strip().upper()
                    if not tck: continue
                    px = get_price(normalize_symbol(tck), mode)
                    rows_out.append((tck, px, r["EPS"], r["BVPS"], r["GN_sheet"], f"App ({mode_badge})"))
                append_history_bulk(now_str, rows_out)
                st.success(f"Snapshot di {len(rows_out)} titoli salvato ✅")

    with tab2:
        # --- lettura + normalizzazione numeri per storicizzazione corretta
        dfh = pd.DataFrame(ws_hist.get_all_records())
        if not dfh.empty:
            dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"], errors="coerce")
            for c in ["Price","EPS","BVPS","Graham","Delta","MarginPct"]:
                if c in dfh.columns:
                    dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
            # calcola Delta/Margin se mancanti
            mask = dfh["Delta"].isna() | dfh["MarginPct"].isna()
            if {"Price","Graham"}.issubset(dfh.columns):
                dfh.loc[mask, "Delta"] = (dfh["Price"] - dfh["Graham"])
                dfh.loc[mask, "MarginPct"] = (1 - (dfh["Price"]/dfh["Graham"])) * 100

            try: current_tick = label_to_ticker[selected_label]
            except: current_tick = None
            dft = dfh[dfh["Ticker"].astype(str).str.upper()==(current_tick or "").upper()] if current_tick else dfh
            dft = dft.sort_values("Timestamp")

            if not dft.empty and pd.notna(dft.iloc[-1].get("Timestamp")):
                st.success(f"✅ Ultimo snapshot: {pd.to_datetime(dft.iloc[-1]['Timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")

            # filtro date
            min_day = pd.to_datetime(dft["Timestamp"]).dt.date.min()
            max_day = pd.to_datetime(dft["Timestamp"]).dt.date.max()
            start,end = st.date_input("Intervallo date", value=(min_day,max_day), min_value=min_day, max_value=max_day)

            if isinstance(start,date) and isinstance(end,date) and start<=end:
                dft = dft[(pd.to_datetime(dft["Timestamp"]).dt.date>=start) &
                          (pd.to_datetime(dft["Timestamp"]).dt.date<=end)]

            # --- Dataframe formattato (prezzi 3 decimali, resto 2)
            show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"] if c in dft.columns]
            disp = dft[show_cols].copy()
            if "Price" in disp:   disp["Price"]   = disp["Price"].map(lambda v: fmt_it(v,3))
            for c in ["EPS","BVPS","Graham","Delta"]:
                if c in disp: disp[c] = disp[c].map(lambda v: fmt_it(v,2))
            if "MarginPct" in disp: disp["MarginPct"] = disp["MarginPct"].map(lambda v: (fmt_it(v,2)+"%") if pd.notnull(v) else "")
            st.dataframe(disp, use_container_width=True, hide_index=True)

            # chart
            if set(["Timestamp","Price","Graham"]).issubset(dft.columns):
                plot_df = dft[["Timestamp","Price","Graham"]].dropna().set_index("Timestamp")
                st.line_chart(plot_df, use_container_width=True)
        else:
            st.info("La tab Storico è vuota.")
