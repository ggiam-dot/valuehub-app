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

# =========================
# CONFIG (da Secrets)
# =========================
SHEET_ID  = st.secrets.get("google_sheet_id")                  # obbligatorio
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

# FTSE MIB stripe (opzionale)
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
# THEME & CSS
# =========================
def inject_theme_css(dark: bool):
    if dark:
        bg="#0e1117"; paper="#161a23"; text="#e6e6e6"; sub="#bdbdbd"
        accent="#4da3ff"; border="#2a2f3a"; good="#9ad17b"; bad="#ff6b6b"; gold="#DAA520"
        metric_val="#f2f2f2"; metric_lab="#cfcfcf"
        formula_bg="#10351e"; formula_border="#2f8f5b"; formula_text="#e6ffef"; stripe_bg="#121723"
        pill_bg="#1e2635"
    else:
        bg="#ffffff"; paper="#fafafa"; text="#222"; sub="#666"
        accent="#0b74ff"; border="#e5e7eb"; good="#0a7f2e"; bad="#b00020"; gold="#DAA520"
        metric_val="#111"; metric_lab="#444"
        formula_bg="#e9f8ef"; formula_border="#b8e6c9"; formula_text="#0d5b2a"; stripe_bg="#f7f7f8"
        pill_bg="#eef2ff"
    st.markdown(f"""
    <style>
    :root {{
      --bg:{bg}; --paper:{paper}; --text:{text}; --sub:{sub};
      --accent:{accent}; --border:{border}; --good:{good}; --bad:{bad}; --gold:{gold};
      --metric-val:{metric_val}; --metric-lab:{metric_lab};
      --formula-bg:{formula_bg}; --formula-border:{formula_border}; --formula-text:{formula_text};
      --stripe-bg:{stripe_bg}; --pill-bg:{pill_bg};
    }}
    .stApp {{ background-color: var(--bg); color: var(--text); }}
    .v-card {{ background: var(--paper); border:1px solid var(--border);
               border-radius:14px; padding:14px 16px; }}
    .v-sub {{ color: var(--sub); font-size:12px; }}
    .v-links {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
    .v-link {{ display:flex; gap:8px; align-items:center; font-size:13px; }}
    .v-link img {{ width:20px; height:20px; }}
    .pill {{ background:var(--pill-bg); padding:4px 10px; border-radius:999px; border:1px solid var(--border); font-weight:600; }}
    .btn-link {{ background:var(--paper); border:1px solid var(--border); padding:6px 10px; border-radius:10px; text-decoration:none; }}
    .btn-link:hover {{ border-color:var(--accent); }}
    .stripe {{
      background: var(--stripe-bg); border: 1px solid var(--border); border-radius: 12px;
      padding: 8px 12px; display:flex; align-items:center; justify-content:space-between; gap: 10px;
      margin-bottom: 10px;
    }}
    .pct-pos {{ color: var(--good); font-weight:700; }}
    .pct-neg {{ color: var(--bad);  font-weight:700; }}
    .v-formula-title {{ font-size: 1.05rem; font-weight:800; margin: 6px 0 8px; }}
    .v-formula-box {{ background: var(--formula-bg); border:1px solid var(--formula-border);
                      border-radius:12px; padding: 12px 14px; color: var(--formula-text); }}
    .v-formula-code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:15px; font-weight:700; }}
    </style>
    """, unsafe_allow_html=True)

tcol1, tcol2 = st.columns([4,1], vertical_alignment="center")
with tcol1: st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")
with tcol2: dark_mode = st.toggle("🌙", value=False, help="Light/Dark mode", label_visibility="collapsed")
inject_theme_css(dark_mode)

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
    # auto
    return price_live(symbol) if is_it_market_open() else price_close(symbol)

# FTSE MIB quote + pct vs close
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
    if changed:
        ws_hist.update(f"A1:I1", [header], value_input_option="USER_ENTERED")

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

# =========================
# LOAD FUNDAMENTALS
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values)<2: return pd.DataFrame(), {}
    header, data = values[0], values[1:]
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
    return df, {"has_name": idx_n>=0, "has_inv": idx_i>=0, "has_morn": idx_m>=0, "has_isin": idx_is>=0}

# =========================
# UI
# =========================
df, meta = load_fundamentals_by_letter()

# --- FTSE MIB STRIPE ---
mib_mode_default = "live" if is_it_market_open() else "close"
mib_last, mib_pct = mib_quote(mib_mode_default)
status = "Aperto" if mib_mode_default=="live" else "Chiuso"
pct_html = "" if mib_pct is None else f"<span class='{'pct-pos' if mib_pct>=0 else 'pct-neg'}'>{mib_pct:+.2f}%</span>"
st.markdown(f"""
<div class="stripe">
  <div class="pill">🇮🇹 FTSE MIB · {status}</div>
  <div style="font-weight:700">{ (f"{mib_last:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")) if mib_last else "n/d" } {pct_html}</div>
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
        return company_name(normalize_symbol(t)) or ""

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

        c_ref1, c_ref2, c_ref3 = st.columns([1.2,1,2])
        with c_ref1:
            price_mode = st.radio("Origine prezzo", ["Auto","Intraday","Chiusura"], horizontal=True, index=0,
                                  help="Auto: intraday in orario di mercato, altrimenti chiusura precedente")
        with c_ref2:
            if st.button("🔄 Aggiorna ora"): st.cache_data.clear(); st.rerun()
        with c_ref3:
            auto = st.toggle("Auto-refresh 60s", value=False, help="Aggiorna automaticamente i prezzi")
            if auto:
                try: st.autorefresh(interval=60_000, key="auto-refresh")
                except: pass

        mode = {"Auto":"auto","Intraday":"live","Chiusura":"close"}[price_mode]
        price_val = get_price(symbol, mode)
        mode_badge = {"auto":"Auto","live":"Intraday","close":"Chiusura"}[mode]

        company = get_display_name(tick)
        isin = str(row.get("ISIN") or "").strip()

        # Link (icone 20px + testo)
        yahoo_url = f"https://finance.yahoo.com/quote/{tick}"
        inv_url   = (row.get("InvestingURL") or "").strip() or f"https://it.investing.com/search/?q={isin or tick}"
        mor_url   = (row.get("MorningURL")  or "").strip() or f"https://www.morningstar.com/search?query={isin or tick}"

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

        # Metriche
        margin_pct = (1 - (price_val/gn_sheet))*100 if (price_val is not None and gn_sheet not in (None,0)) else None
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric(f"Prezzo ({mode_badge})", f"{price_val:.2f}" if price_val is not None else "n/d")
        with c2: st.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
        with c3:
            if margin_pct is not None:
                label = "Sottovalutata" if margin_pct>0 else "Sopravvalutata"
                st.metric("Margine", f"{margi
