import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
from datetime import datetime, date
import json

import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
# Inserisci nei Secrets:
# google_sheet_id = "10AZY-ePyRssx6ajoF6_hsFChMK1dWVA-wkpDdz-DyM8"
# fund_tab = "Fondamentali"
# hist_tab = "Storico"
# yf_suffix = ".MI"
SHEET_ID  = st.secrets.get("google_sheet_id", "10AZY-ePyRssx6ajoF6_hsFChMK1dWVA-wkpDdz-DyM8")
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

st.set_page_config(page_title="Value Hub – Graham Lookup", page_icon="📈", layout="centered")
st.title("📈 Value Investment – Graham Lookup")
st.caption("EPS e BVPS letti dal foglio 'Fondamentali'. Prezzo live via Yahoo Finance. Snapshot EOD su 'Storico'.")

# =========================
# GOOGLE AUTH
# =========================
@st.cache_resource
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    # Preferisci secrets su Streamlit Cloud
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
    else:
        # fallback locale: file service_account.json nella stessa cartella
        with open("service_account.json","r") as f:
            creds_dict = json.load(f)
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

gc = get_gsheet_client()
# ✅ Apriamo direttamente per ID, niente ricerca per nome
sh = gc.open_by_key(SHEET_ID)
ws_fund = sh.worksheet(FUND_TAB)
ws_hist = sh.worksheet(HIST_TAB)

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals():
    df = pd.DataFrame(ws_fund.get_all_records())
    for col in ["Ticker","EPS","BVPS"]:
        if col not in df.columns:
            df[col] = np.nan
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        return ""
    return s if "." in s else s + YF_SUFFIX

@st.cache_data(show_spinner=False)
def fetch_price_yf(symbol: str):
    try:
        t = yf.Ticker(symbol)
        price = None
        fi = getattr(t, "fast_info", None)
        if fi:
            price = fi.get("last_price")
        if price is None:
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
        if price is None:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

def graham_number(eps, bvps):
    try:
        eps = float(eps); bvps = float(bvps)
        if eps <= 0 or bvps <= 0: return None
        return sqrt(22.5 * eps * bvps)
    except:
        return None

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    row = [ts, ticker, price if price is not None else "", eps, bvps, graham if graham is not None else "", fonte]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def last_eod_for_ticker(ticker: str):
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty or "Ticker" not in dfh.columns or "Timestamp" not in dfh.columns:
        return None
    dft = dfh[dfh["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if dft.empty:
        return None
    dft["Timestamp"] = pd.to_datetime(dft["Timestamp"], errors="coerce")
    dft = dft.sort_values("Timestamp")
    return dft.iloc[-1].to_dict()

# =========================
# UI
# =========================
df = load_fundamentals()
if df.empty or df["Ticker"].isna().all():
    st.warning("Nessun ticker in 'Fondamentali'. Aggiungi almeno una riga con Ticker, EPS e BVPS.")
else:
    tickers = df["Ticker"].dropna().tolist()
    tick = st.selectbox("Scegli il Ticker", options=tickers)

    if tick:
        rec = df.loc[df["Ticker"] == tick].iloc[0].to_dict()
        eps  = rec.get("EPS", None)
        bvps = rec.get("BVPS", None)

        # Stato EOD
        eod = last_eod_for_ticker(tick)
        if eod and eod.get("Timestamp"):
            ts = pd.to_datetime(eod["Timestamp"])
            is_today = (ts.date() == date.today())
            if is_today:
                st.success(f"✅ Dati EOD presenti: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning(f"⚠️ Ultimo EOD: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("ℹ️ Nessuno snapshot EOD trovato per questo ticker.")

        symbol = normalize_symbol(tick)
        price_live = fetch_price_yf(symbol)
        gnum = graham_number(eps, bvps)

        st.subheader(f"📌 {tick}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
            st.caption(f"Fonte prezzo: Yahoo Finance – {symbol}")
        with c2:
            st.metric("Numero di Graham", f"{gnum:.2f}" if gnum is not None else "n/d")
            st.caption("Formula: √(22.5 × EPS × BVPS)")

        # Formula esplicita
        st.markdown("### Dettaglio calcolo Graham")
        if gnum is not None and eps is not None and bvps is not None:
            try:
                st.code(f"√(22.5 × {float(eps):.4f} × {float(bvps):.4f}) = {gnum:.4f}")
            except Exception:
                st.code("√(22.5 × EPS × BVPS)")
        else:
            st.write("Numero di Graham non calcolabile: controlla che EPS e BVPS siano > 0.")

        # Margine di sicurezza 67%
        if price_live is not None and gnum is not None:
            threshold = 0.67 * gnum
            ok = price_live <= threshold
            st.markdown(f"**Margine di sicurezza (67%)**: {'✅ SÌ' if ok else '❌ NO'} — soglia {threshold:.2f}")

        if st.button("💾 Salva snapshot su 'Storico'"):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_history_row(now_str, tick, price_live, eps, bvps, gnum, "App")
            st.success("Snapshot salvato su 'Storico'.")
