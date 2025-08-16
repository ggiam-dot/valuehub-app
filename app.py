import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
from datetime import datetime, date
import json, re
import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG (da Secrets)
# =========================
SHEET_ID  = st.secrets.get("google_sheet_id")                  
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

# Lettere di colonna
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")

st.set_page_config(page_title="Vigil – Value_Investment_Graham_Intelligent_Lookup", 
                   page_icon="📈", layout="centered")
st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")

# =========================
# SIDEBAR: Admin/Public toggle
# =========================
is_admin = st.sidebar.toggle("🔑 Modalità Admin", value=True)

# =========================
# GOOGLE AUTH
# =========================
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
# UTILITIES
# =========================
def _letter_to_index(letter: str) -> int:
    s = letter.strip().upper()
    n = 0
    for ch in s:
        n = n*26 + (ord(ch)-64)
    return n-1

def to_number(x):
    if x is None: return None
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace("\u00A0","").replace("€","").replace("%","")
    s = s.replace("\u2212","-")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): s = s.replace(".","").replace(",",".")
        else: s = s.replace(",","")
    elif "," in s: s = s.replace(",",".")
    try: return float(s)
    except: return None

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

@st.cache_data(show_spinner=False)
def fetch_price_yf(symbol: str):
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t,"fast_info",None)
        price = fi.get("last_price") if fi else None
        if not price:
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
        if not price:
            h = t.history(period="5d")
            if not h.empty: price = float(h["Close"].iloc[-1])
        return float(price) if price else None
    except: return None

def gn_formula_225(eps, bvps):
    if eps and bvps and eps>0 and bvps>0:
        return sqrt(22.5*eps*bvps)
    return None

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App", margin=None, delta=None):
    row = [ts, ticker, price or "", eps, bvps, graham or "", delta or "", margin or "", fonte]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

# =========================
# LOAD FUNDAMENTALS
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    header, data = values[0], values[1:]
    idx_t, idx_e, idx_b, idx_g = map(_letter_to_index,[TICKER_LETTER,EPS_LETTER,BVPS_LETTER,GN_LETTER])
    df = pd.DataFrame({
        "Ticker":[row[idx_t] for row in data],
        "EPS":[to_number(row[idx_e]) for row in data],
        "BVPS":[to_number(row[idx_b]) for row in data],
        "GN_sheet":[to_number(row[idx_g]) for row in data],
    })
    df = df[df["Ticker"].astype(str).str.strip()!=""].reset_index(drop=True)
    return df

df = load_fundamentals_by_letter()
tickers = df["Ticker"].dropna().unique().tolist()

# =========================
# UI – Tabs
# =========================
tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

with tab1:
    tick = st.selectbox("Scegli il Ticker / Nome società", sorted(tickers))
    if tick:
        row = df[df["Ticker"]==tick].iloc[0]
        eps, bvps, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_formula = gn_formula_225(eps,bvps)
        price = fetch_price_yf(normalize_symbol(tick))

        margin_pct = None
        if price and gn_sheet: margin_pct = (1-(price/gn_sheet))*100

        c1,c2,c3 = st.columns(3)
        c1.metric("Prezzo live", f"{price:.2f}" if price else "n/d")
        c2.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet else "n/d")
        c3.metric("Margine", f"{margin_pct:.2f}%" if margin_pct else "n/d")

        st.markdown("### The GN Formula")
        if gn_formula: st.code(f"√(22.5 × {eps:.4f} × {bvps:.4f}) = {gn_formula:.4f}")

        if is_admin:
            st.markdown("---")
            cb1, cb2 = st.columns(2)
            with cb1:
                if st.button("🔄 Aggiorna dal foglio"):
                    st.cache_data.clear(); st.rerun()
            with cb2:
                if st.button("✍️ Riscrivi Graham# su Sheet"):
                    gn_series = df.apply(lambda r: gn_formula_225(r["EPS"],r["BVPS"]),axis=1)
                    out = [[v if v else ""] for v in gn_series]
                    ws_fund.update(f"{GN_LETTER}2:{GN_LETTER}{len(out)+1}", out)
                    st.success("Aggiornato Graham# su Sheet")
                    st.cache_data.clear(); st.rerun()

with tab2:
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if not dfh.empty:
        dft = dfh[dfh["Ticker"].str.upper()==tick.upper()] if tick else dfh
        dft["Timestamp"] = pd.to_datetime(dft["Timestamp"])
        dft = dft.sort_values("Timestamp")

        # notifica ultimo snapshot SOLO QUI
        if not dft.empty:
            st.success(f"✅ Ultimo snapshot: {dft.iloc[-1]['Timestamp']}")

        min_day,max_day = dft["Timestamp"].dt.date.min(), dft["Timestamp"].dt.date.max()
        start,end = st.date_input("Intervallo date",(min_day,max_day),min_value=min_day,max_value=max_day)
        if start and end:
            dft = dft[(dft["Timestamp"].dt.date>=start)&(dft["Timestamp"].dt.date<=end)]

        st.dataframe(dft, use_container_width=True, hide_index=True)

        csv = dft.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica CSV", data=csv, file_name=f"storico_{tick}.csv", mime="text/csv")

# Debug in fondo, solo admin
if is_admin:
    st.markdown("---")
    with st.expander("🔎 Debug"):
        st.write(df)
