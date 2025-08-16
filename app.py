import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
from datetime import datetime
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

# Lettere colonne
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Vigil – Value Investment Graham Lookup",
                   page_icon="📈", layout="centered")
st.title("📈 Vigil – Value Investment Graham Lookup")

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
    s = str(x).strip().replace("\u00A0","").replace("€","").replace("EUR","")
    s = s.replace("\u2212","-").replace("%","")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if s in {"", "-", ","}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
    else:
        if "," in s: s = s.replace(",",".")
    try: return float(s)
    except: return None

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

@st.cache_data(show_spinner=False)
def fetch_price_yf(symbol: str):
    try:
        t = yf.Ticker(symbol)
        price = None
        fi = getattr(t, "fast_info", None)
        if fi: price = fi.get("last_price")
        if price is None:
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
        if price is None:
            h = t.history(period="5d")
            if not h.empty: price = float(h["Close"].iloc[-1])
        return float(price) if price is not None else None
    except: return None

def gn_formula_225(eps,bvps):
    if not eps or not bvps or eps<=0 or bvps<=0: return None
    return sqrt(22.5*eps*bvps)

def append_history_row(ts,ticker,price,eps,bvps,gn,fonte="App"):
    margin = (1-(price/gn))*100 if price and gn else None
    row = [ts,ticker,price or "",eps,bvps,gn or "",margin or "",fonte]
    ws_hist.append_row(row,value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values)<2: return pd.DataFrame(),{}
    header, data = values[0], values[1:]
    idx_t = _letter_to_index(TICKER_LETTER)
    idx_e = _letter_to_index(EPS_LETTER)
    idx_b = _letter_to_index(BVPS_LETTER)
    idx_g = _letter_to_index(GN_LETTER)
    df = pd.DataFrame({
        "Ticker":[r[idx_t] for r in data if len(r)>idx_t],
        "EPS":[to_number(r[idx_e]) if len(r)>idx_e else None for r in data],
        "BVPS":[to_number(r[idx_b]) if len(r)>idx_b else None for r in data],
        "GN_sheet":[to_number(r[idx_g]) if len(r)>idx_g else None for r in data],
    })
    df = df[df["Ticker"].astype(str).str.strip()!=""].reset_index(drop=True)
    return df, {"header_row":header}

# =========================
# APP BODY
# =========================
df,_ = load_fundamentals_by_letter()
if df.empty:
    st.warning("Nessun dato utile.")
else:
    # ricerca anche per nome
    options = df["Ticker"].tolist()
    tick = st.selectbox("Scegli il Ticker o Nome Società", options=options)

    if tick:
        row = df[df["Ticker"]==tick].iloc[0]
        eps,bvps,gn_sheet = row["EPS"],row["BVPS"],row["GN_sheet"]
        gn_calc = gn_formula_225(eps,bvps)
        price = fetch_price_yf(normalize_symbol(tick))
        margin_pct = (1-(price/gn_sheet))*100 if price and gn_sheet else None

        c1,c2,c3 = st.columns(3)
        c1.metric("Prezzo live", f"{price:.2f}" if price else "n/d")
        c2.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet else "n/d")
        c3.metric("Margine", f"{margin_pct:.2f}%" if margin_pct else "n/d")

        st.markdown("### The GN Formula")
        if gn_calc: st.code(f"√(22.5 × {eps:.4f} × {bvps:.4f}) = {gn_calc:.4f}")
        else: st.write("Formula non calcolabile.")

        st.markdown("---")
        # Toggle admin in basso
        is_admin = st.toggle("🛠️ Modalità amministratore", value=False,
                             help="Mostra/nasconde i comandi admin")
        if is_admin:
            col1,col2,col3 = st.columns(3)
            with col1:
                if st.button("🔄 Aggiorna", use_container_width=True):
                    st.cache_data.clear(); st.rerun()
            with col2:
                if st.button("✍️ Riscrivi Graham#", use_container_width=True):
                    gn_series = df.apply(lambda r: gn_formula_225(r["EPS"],r["BVPS"]),axis=1)
                    out = [[("" if pd.isna(v) else float(v))] for v in gn_series]
                    start_row,end_row=2,1+len(out)
                    ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}",out,
                                   value_input_option="USER_ENTERED")
                    st.success("Colonna Graham# aggiornata.")
                    st.cache_data.clear(); st.rerun()
            with col3:
                if st.button("💾 Salva snapshot", use_container_width=True):
                    now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_history_row(now,tick,price,eps,bvps,gn_sheet,"App")
                    st.success("Snapshot salvato.")
