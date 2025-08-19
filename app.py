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
from streamlit_autorefresh import st_autorefresh

# ============== CONFIG ==============
SHEET_ID  = st.secrets.get("google_sheet_id")
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

MIB_SYMBOLS = [s.strip() for s in st.secrets.get(
    "mib_symbols", "^FTSEMIB,FTSEMIB.MI,FTMIB.MI,MIB.MI"
).split(",") if s.strip()]
BORSA_LINK = st.secrets.get("borsa_link", "https://www.borsaitaliana.it/")

TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")
NAME_LETTER   = st.secrets.get("name_col_letter", "")
INV_URL_LETTER= st.secrets.get("investing_col_letter", "")
MORN_URL_LETTER=st.secrets.get("morning_col_letter", "")
ISIN_LETTER   = st.secrets.get("isin_col_letter", "")

PRICE_TTL = int(st.secrets.get("price_ttl_seconds", 20))
AUTOREFRESH_MS = int(st.secrets.get("autorefresh_ms", 20000))

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup",
                   page_icon="📈", layout="wide")

# ============== THEME/CSS (chiaro + dark) ==============
if "dark" not in st.session_state:
    st.session_state.dark = False

def inject_css(dark: bool):
    if dark:
        bg="#0b0f16"; paper="#131a24"; text="#f3f7fb"; sub="#c6cfdb"; border="#2a3340"
        good="#0abf53"; bad="#ff4d4f"; gold="#FFD166"; accent="#66b1ff"
        chip="#1b2432"; strip="#0f1320"; df_text="#f3f7fb"
    else:
        bg="#ffffff"; paper="#ffffff"; text="#151515"; sub="#4a4a4a"; border="#222"
        good="#0a7f2e"; bad="#c62828"; gold="#C79200"; accent="#0a66ff"
        chip="#eef3ff"; strip="#f7f9fc"; df_text="#111"

    st.markdown(f"""
    <style>
      :root {{
        --bg:{bg}; --paper:{paper}; --text:{text}; --sub:{sub}; --border:{border};
        --good:{good}; --bad:{bad}; --gold:{gold}; --accent:{accent}; --chip:{chip}; --strip:{strip}; --dftext:{df_text};
      }}
      .stApp{{ background:var(--bg); color:var(--text); }}
      .v-title{{ font-weight:800; font-size:1.28rem; margin:0 0 8px; }}
      .v-title .v-light{{ font-weight:700; }}
      .v-card{{ background:var(--paper); border:1px solid #ddd; border-radius:14px; padding:12px 14px; }}
      .v-sub{{ color:var(--sub); font-size:12px; }}
      .v-links{{ display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
      .v-link{{ display:flex; gap:8px; align-items:center; font-size:14px; color:var(--text); }}
      .v-link img{{ width:20px; height:20px; }}
      .stripe{{ background:var(--strip); border:1px solid #ddd; border-radius:12px; padding:10px 12px;
               display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
      .pill{{ background:var(--chip); color:var(--text); border:1px solid #cfd6e1; padding:6px 12px; border-radius:999px; font-weight:800; }}
      .pct-pos{{ color:var(--good); font-weight:900; }}
      .pct-neg{{ color:var(--bad);  font-weight:900; }}

      /* Metriche leggibili + label grigio scuro */
      [data-testid="stMetric"] > div > div:nth-child(1){{ color:#333 !important; font-weight:800; }}
      [data-testid="stMetricValue"]{{ color:var(--text) !important; font-weight:900; }}

      .judge-box{{ border:2px solid #ddd; border-radius:14px; padding:12px; text-align:center; }}
      .judge-box.good{{ border-color:var(--good); }}
      .judge-box.bad{{  border-color:var(--bad);  }}
      .judge-val{{ font-size:28px; font-weight:900; }}
      .judge-val.good{{ color:var(--good); }}
      .judge-val.bad{{  color:var(--bad);  }}
      .judge-lbl{{ margin-top:6px; font-size:16px; font-weight:900; }}
      .judge-lbl.good{{ color:var(--good); }}
      .judge-lbl.bad{{  color:var(--bad);  }}
      .judge-lbl .gstar{{ color:var(--gold); margin-left:6px; }}

      .stButton>button{{ background:var(--paper); color:var(--text); border:1px solid var(--border);
                        border-radius:12px; padding:8px 14px; font-weight:800; }}
      .stButton>button:hover{{ border-color:var(--accent); }}

      /* mini refresh icona accanto al prezzo */
      .mini-btn>button{{ width:44px !important; padding:6px 8px !important; border-radius:8px !important; font-weight:900 !important; }}

      /* tab & dataframe testo scuro/chiaro */
      .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] div{{ color:var(--text) !important; }}
      .stDataFrame, .stDataFrame *{{ color:var(--dftext) !important; }}
    </style>
    """, unsafe_allow_html=True)

hdr_l, hdr_r = st.columns([4,1], vertical_alignment="center")
with hdr_l:
    st.markdown("<div class='v-title'>📈 Vigil – Value Investment Graham <span class='v-light'>(Intelligent)</span> Lookup</div>", unsafe_allow_html=True)
with hdr_r:
    st.session_state.dark = st.toggle("🌙 Dark", value=st.session_state.dark)
inject_css(st.session_state.dark)

# ============== TIME/AUTH ==============
ROME = ZoneInfo("Europe/Rome")
def market_open(now=None):
    now = now or datetime.now(ROME)
    return now.weekday() < 5 and dtime(9,0) <= now.time() <= dtime(17,35)

@st.cache_resource
def gsheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets.get("gcp_service_account")
    if not creds_dict:
        with open("service_account.json","r") as f: creds_dict = json.load(f)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = gsheet()
sh = gc.open_by_key(SHEET_ID)
ws_fund = sh.worksheet(FUND_TAB)
ws_hist = sh.worksheet(HIST_TAB)

# ============== UTILS ==============
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")

def _letter_to_index(letter: str) -> int:
    if not letter: return -1
    s = letter.strip().upper(); n=0
    for ch in s:
        if not ("A" <= ch <= "Z"): return -1
        n = n*26 + (ord(ch)-64)
    return n-1

def parse_num(s):
    """Robusto: '1.234,56' | '1,234.56' | '75,48000336' | '75.48000336' ->  float"""
    if s is None or s == "": return np.nan
    if isinstance(s, (int, float)): 
        try: return float(s)
        except: return np.nan
    s = str(s).strip().replace("\u00A0","")
    s = re.sub(r"[^\d,.\-+]", "", s)
    if s.count(",") and s.count("."):
        last = max(s.rfind(","), s.rfind("."))
        dec = s[last]
        s = s.replace("," if dec=="." else ".", "")
        s = s.replace(dec, ".")
    elif "," in s:
        s = s.replace(",", ".")
    try: return float(s)
    except: return np.nan

def to_number_sheet(s):
    v = parse_num(s)
    return None if (isinstance(v,float) and np.isnan(v)) else float(v)

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

def fmt_it(x, dec=2):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "n/d" if dec==3 else ""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Prezzi
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

@st.cache_data(ttl=600, show_spinner=False)
def last_close(symbol: str):
    try:
        h = yf.Ticker(symbol).history(period="10d", interval="1d")["Close"].dropna()
        if len(h)>0: return float(h.iloc[-1])
    except: pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def prev_close(symbol: str):
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t,"fast_info",None)
        pc = fi.get("previous_close") if fi else None
        if pc: return float(pc)
        h = t.history(period="10d", interval="1d")["Close"].dropna()
        if len(h)>=2: return float(h.iloc[-2])
    except: pass
    return None

def auto_price(symbol: str):
    return price_live(symbol) if market_open() else last_close(symbol)

@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def mib_summary():
    last_c = prev_c = live = None
    for sym in MIB_SYMBOLS:
        try:
            h = yf.Ticker(sym).history(period="10d", interval="1d")["Close"].dropna()
            if len(h)>=1 and last_c is None: last_c = float(h.iloc[-1])
            if len(h)>=2 and prev_c is None: prev_c = float(h.iloc[-2])
        except: pass
    for sym in MIB_SYMBOLS:
        try:
            t = yf.Ticker(sym)
            fi = getattr(t,"fast_info",None)
            if fi and fi.get("last_price"):
                live = float(fi.get("last_price")); break
            intr = t.history(period="1d", interval="1m")["Close"].dropna()
            if len(intr)>0: live = float(intr.iloc[-1]); break
        except: pass
    return {"live": live, "last_close": last_c, "prev_close": prev_c}

@st.cache_data(ttl=86400, show_spinner=False)
def company_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info
        return str(info.get("shortName") or info.get("longName") or "")
    except: return ""

def gn_225(eps,bvps):
    if eps and bvps and eps>0 and bvps>0: return sqrt(22.5*eps*bvps)
    return None

# Storico helpers
DESIRED_HIST_HEADER = ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"]

def ensure_history_headers():
    values = ws_hist.get_all_values()
    if not values:
        ws_hist.insert_row(DESIRED_HIST_HEADER,1); return
    header = values[0]; changed=False
    for col in DESIRED_HIST_HEADER:
        if col not in header: header.append(col); changed=True
    if changed: ws_hist.update("A1:I1", [header], value_input_option="USER_ENTERED")

def normalize_history_headers_strict():
    ws_hist.update("A1:I1", [DESIRED_HIST_HEADER], value_input_option="USER_ENTERED")

@st.cache_data(ttl=60, show_spinner=False)
def load_history_df():
    """Legge VALORI GREZZI dal foglio (niente casting automatico)."""
    rows = ws_hist.get_all_values()
    if not rows: return pd.DataFrame()
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    # parse numeri robusto
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for c in ["Price","EPS","BVPS","Graham","Delta","MarginPct"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_num)
    # completa Delta/Margin se mancanti
    if {"Price","Graham"}.issubset(df.columns):
        mask = df["Delta"].isna() | df["MarginPct"].isna()
        df.loc[mask, "Delta"] = (df["Price"] - df["Graham"])
        df.loc[mask, "MarginPct"] = (1 - (df["Price"]/df["Graham"])) * 100
    return df

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

# ============== LOAD FONDAMENTALI ==============
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
    df["EPS"]      = df["EPS_raw"].apply(to_number_sheet)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number_sheet)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number_sheet)
    df["Name"]     = (df["Name_raw"].astype(str).str.strip() if isinstance(df["Name_raw"], pd.Series) else "")
    df["InvestingURL"] = (df["InvestingURL_raw"].astype(str).str.strip() if isinstance(df["InvestingURL_raw"], pd.Series) else "")
    df["MorningURL"]   = (df["MorningURL_raw"].astype(str).str.strip() if isinstance(df["MorningURL_raw"], pd.Series) else "")
    df["ISIN"]         = (df["ISIN_raw"].astype(str).str.strip() if isinstance(df["ISIN_raw"], pd.Series) else "")
    return df, {}

# ============== UI ==============
df, _ = load_fundamentals_by_letter()

# FTSE MIB stripe
mib = mib_summary()
open_now = market_open()
main_val = mib["live"] if (open_now and mib["live"] is not None) else mib["last_close"]
base = mib["prev_close"]
pct = None if (main_val is None or base is None or base==0) else ((main_val - base)/base)*100
status = "Aperto" if open_now else "Chiuso"
pct_html = "" if pct is None else f"<span class='{'pct-pos' if pct>=0 else 'pct-neg'}'>{pct:+.2f}%</span>"
sub = f"Ultima chiusura: {fmt_it(mib['last_close'],3)} • Precedente: {fmt_it(mib['prev_close'],3)}" if mib["last_close"] and mib["prev_close"] else "Ultima chiusura non disponibile"

st.markdown(f"""
<div class="stripe">
  <div class="pill">🇮🇹 FTSE MIB · {status}</div>
  <div style="font-weight:900">{ fmt_it(main_val,3) } {pct_html}</div>
  <div class="v-sub">{sub}</div>
  <div><a class="v-link" href="{BORSA_LINK}" target="_blank" rel="noopener">Borsa Italiana ↗︎</a></div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("Nessun dato utile. Controlla il foglio.")
else:
    st_autorefresh(interval=AUTOREFRESH_MS, key="auto-refresh")  # refresh automatico

    @st.cache_data(show_spinner=False, ttl=86400)
    def display_name(t: str) -> str:
        r = df[df["Ticker"]==t]
        if not r.empty:
            n = str(r.iloc[0].get("Name") or "").strip()
            if n: return n
        try:
            info = yf.Ticker(normalize_symbol(t)).info
            return str(info.get("shortName") or info.get("longName") or "")
        except: return ""

    tickers_all = sorted(df["Ticker"].tolist())
    label_to_ticker = { (f"{t} — {display_name(t)}" if display_name(t) else t): t for t in tickers_all }
    tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

    # ----- TAB ANALISI
    with tab1:
        selected_label = st.selectbox("Scegli il Ticker", options=list(label_to_ticker.keys()), index=0)
        tick = label_to_ticker[selected_label]
        row = df[df["Ticker"]==tick].iloc[0]
        eps_val, bvps_val, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_applied = gn_225(eps_val, bvps_val)
        symbol = normalize_symbol(tick)

        isin_raw = str(row.get("ISIN") or "").strip().upper()
        isin = isin_raw if ISIN_RE.match(isin_raw) else ""
        q = isin if isin else tick
        yahoo_url = f"https://finance.yahoo.com/quote/{tick}"
        inv_url   = (row.get("InvestingURL") or "").strip() or f"https://it.investing.com/search/?q={q}"
        mor_url   = (row.get("MorningURL")  or "").strip() or f"https://www.morningstar.com/search?query={q}"

        st.markdown(f"""
        <div class="v-card" style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
          <div>
            <h3 style="margin:0">{tick} — {display_name(tick)}</h3>
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

        # PREZZO + mini refresh a destra + GN + judge box
        px = auto_price(symbol)
        pc = prev_close(symbol)
        delta_pct = None if (px is None or pc in (None,0)) else ((px - pc)/pc*100)

        col_price, col_refresh, col_gn, col_judge = st.columns([1.5, 0.12, 1.1, 1.2], vertical_alignment="center")
        with col_price:
            st.metric("Prezzo", ("€ " + fmt_it(px,3)) if px is not None else "n/d",
                      (f"{delta_pct:+.2f}%" if delta_pct is not None else None))
        with col_refresh:
            st.markdown("<div class='mini-btn' style='display:flex;justify-content:flex-start;'>", unsafe_allow_html=True)
            if st.button("⟳", key="refresh_now"):
                st.cache_data.clear(); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with col_gn:
            st.metric("Graham#", ("€ " + fmt_it(gn_sheet,2)) if gn_sheet is not None else "n/d")
        with col_judge:
            margin_pct = (1 - (px/gn_sheet))*100 if (px is not None and gn_sheet not in (None,0)) else None
            if margin_pct is None:
                st.metric("Margine", "n/d")
            else:
                cls = "good" if margin_pct>0 else "bad"
                star = " <span class='gstar'>⭐G</span>" if margin_pct >= 33 else ""
                st.markdown(
                    f"<div class='judge-box {cls}'>"
                    f"<div class='judge-val {cls}'>{fmt_it(margin_pct,2)}%</div>"
                    f"<div class='judge-lbl {cls}'>{'Sottovalutata' if margin_pct>0 else 'Sopravvalutata'}{star}</div>"
                    f"</div>", unsafe_allow_html=True
                )

        # Formula
        st.markdown("<div style='margin-top:8px;font-weight:800;color:var(--good)'>The GN Formula (Applied)</div>", unsafe_allow_html=True)
        if gn_applied is not None:
            st.markdown(
                f"<div class='v-card' style='background:#e9f7ef;border-color:#b8e6c9;'>"
                f"<div style='font-family: ui-monospace, Menlo, Consolas, monospace; font-size:17px; font-weight:800;'>"
                f"√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_applied:.4f}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='v-card'>√(22.5 × EPS × BVPS)</div>", unsafe_allow_html=True)

        st.markdown("---")
        is_admin = st.toggle("🛠️ Modalità amministratore", value=True)
        if is_admin:
            c1,c2,c3,c4 = st.columns(4)
            if c1.button("Aggiorna dal foglio", use_container_width=True):
                st.cache_data.clear(); st.rerun()
            if c2.button("Riscrivi Graham# su Sheet", use_container_width=True):
                gn_series = df.apply(lambda r: gn_225(r["EPS"], r["BVPS"]), axis=1)
                out = [[("" if (v is None or pd.isna(v)) else float(v))] for v in gn_series]
                start_row = 2; end_row = start_row + len(out) - 1
                ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}", out, value_input_option="USER_ENTERED")
                st.success("Graham# aggiornato."); st.cache_data.clear(); st.rerun()
            if c3.button("Salva snapshot", use_container_width=True):
                now_str = datetime.now(ROME).strftime("%Y-%m-%d %H:%M:%S")
                append_history_row(now_str, tick, px, eps_val, bvps_val, gn_sheet, "App (auto)")
                st.success("Snapshot salvato.")
            if c4.button("Snapshot TUTTI", use_container_width=True):
                now_str = datetime.now(ROME).strftime("%Y-%m-%d %H:%M:%S")
                for _,r in df.iterrows():
                    tck = str(r["Ticker"]).strip().upper()
                    if not tck: continue
                    append_history_row(now_str, tck, auto_price(normalize_symbol(tck)),
                                       r["EPS"], r["BVPS"], r["GN_sheet"], "App (auto)")
                st.success("Snapshot completo salvato ✅")

    # ----- TAB STORICO (parsing su valori grezzi)
    with tab2:
        if st.button("🧹 Normalizza intestazioni 'Storico' (A1:I1)"):
            normalize_history_headers_strict()
            st.success("Header normalizzato. Ricarico…")
            st.cache_data.clear(); st.rerun()

        dfh = load_history_df()

        if not dfh.empty:
            # filtra su ticker corrente se presente
            try:
                current_tick = label_to_ticker[selected_label]
            except:
                current_tick = None
            dft = dfh[dfh["Ticker"].astype(str).str.upper()==(current_tick or "").upper()] if current_tick else dfh
            dft = dft.sort_values("Timestamp")

            if not dft.empty and pd.notna(dft.iloc[-1].get("Timestamp")):
                st.success(f"✅ Ultimo snapshot: {pd.to_datetime(dft.iloc[-1]['Timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")

            min_day = pd.to_datetime(dft["Timestamp"]).dt.date.min()
            max_day = pd.to_datetime(dft["Timestamp"]).dt.date.max()
            start,end = st.date_input("Intervallo date", value=(min_day,max_day), min_value=min_day, max_value=max_day)
            if isinstance(start,date) and isinstance(end,date) and start<=end:
                dft = dft[(pd.to_datetime(dft["Timestamp"]).dt.date>=start) &
                          (pd.to_datetime(dft["Timestamp"]).dt.date<=end)]

            show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"] if c in dft.columns]
            disp = dft[show_cols].copy()
            if "Price" in disp:   disp["Price"]   = disp["Price"].map(lambda v: "€ "+fmt_it(v,3) if pd.notnull(v) else "")
            for c in ["EPS","BVPS","Graham","Delta"]:
                if c in disp: disp[c] = disp[c].map(lambda v: fmt_it(v,2) if pd.notnull(v) else "")
            if "MarginPct" in disp: disp["MarginPct"] = disp["MarginPct"].map(lambda v: (fmt_it(v,2)+"%") if pd.notnull(v) else "")

            st.dataframe(disp, use_container_width=True, hide_index=True)

            if set(["Timestamp","Price","Graham"]).issubset(dft.columns):
                plot_df = dft[["Timestamp","Price","Graham"]].dropna().set_index("Timestamp")
                st.line_chart(plot_df, use_container_width=True)
        else:
            st.info("La tab Storico è vuota.")
