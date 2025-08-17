import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup", layout="centered")

# ---------- Config & Secrets ----------
APP_PUBLIC = st.secrets["app"].get("public_mode", True)
ADMIN_CODE = st.secrets["app"].get("admin_access_code", "")
DEFAULT_SUFFIX = st.secrets["app"].get("default_suffix", ".MI")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_client_and_ws():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["gsheet"]["sheet_id"])
    fond = sh.worksheet(st.secrets["gsheet"]["fundamentals_tab"])
    hist = sh.worksheet(st.secrets["gsheet"]["history_tab"])
    return gc, sh, fond, hist

@st.cache_data(ttl=300)
def load_fundamentals():
    _, _, fond, _ = get_client_and_ws()
    rows = fond.get_all_records()
    df = pd.DataFrame(rows)
    # Normalize expected columns
    for c in ["Ticker", "EPS", "BVPS"]:
        if c not in df.columns: df[c] = np.nan
    # If 'Graham' missing, compute from EPS/BVPS
    if "Graham" not in df.columns:
        df["Graham"] = np.sqrt(22.5 * df["EPS"].astype(float).clip(lower=0) * df["BVPS"].astype(float).clip(lower=0))
    # Optional columns pass-through
    for c in ["Name", "InvestingURL", "MorningstarURL", "ISIN"]:
        if c not in df.columns: df[c] = ""
    return df

def latest_price(ticker: str):
    t = yf.Ticker(ticker)
    price = None
    # fast path
    try:
        price = float(t.fast_info.last_price)
        if price and price > 0: return price
    except Exception:
        pass
    # fallback
    try:
        hist = t.history(period="1d")
        if len(hist) > 0:
            return float(hist["Close"][-1])
    except Exception:
        pass
    return None

def compute_gn(eps, bvps):
    try:
        eps = float(eps); bvps = float(bvps)
        if eps > 0 and bvps > 0:
            return float(np.sqrt(22.5 * eps * bvps))
    except Exception:
        pass
    return None

def compute_mos(gn, price):
    if gn and gn > 0 and price is not None:
        return (gn - price) / gn
    return None

def mos_star(mos):
    return "⭐️" if mos is not None and mos >= 0.33 else ""

def can_write_actions():
    # Protect write actions (append on Storico)
    code = st.text_input("Admin code per azioni di scrittura", type="password", key="write_code")
    return (code == ADMIN_CODE) and (ADMIN_CODE != "")

def append_history(hist_ws, row_list):
    # Row format expected by your Storico header
    hist_ws.append_row(row_list, value_input_option="USER_ENTERED")

# ---------- UI ----------
st.title("Vigil – Value Investment Graham Intelligent Lookup")
st.caption("Prezzo live (Yahoo), Numero di Graham, Margine di Sicurezza e snapshot su Google Sheets.")

df = load_fundamentals()
tickers = sorted([t for t in df["Ticker"].dropna().astype(str).unique() if t.strip()])

col1, col2 = st.columns([3, 2])
with col1:
    sel = st.selectbox("Seleziona Ticker", options=[""] + tickers, index=0, help="Dai dati del foglio 'Fondamentali'")
with col2:
    manual = st.text_input("…oppure digita un ticker", placeholder="es. ENEL.MI")

ticker = manual.strip() or sel.strip()

if ticker:
    # Recupera riga dal DF (se esiste)
    row = df[df["Ticker"].astype(str) == ticker].head(1)
    eps = float(row["EPS"].iloc[0]) if not row.empty and pd.notnull(row["EPS"].iloc[0]) else None
    bvps = float(row["BVPS"].iloc[0]) if not row.empty and pd.notnull(row["BVPS"].iloc[0]) else None
    gn_sheet = float(row["Graham"].iloc[0]) if not row.empty and pd.notnull(row["Graham"].iloc[0]) else None

    price = latest_price(ticker)
    gn_calc = compute_gn(eps, bvps)
    gn = gn_sheet if gn_sheet else gn_calc
    mos = compute_mos(gn, price)

    st.subheader(f"{ticker}")
    if not row.empty and row["Name"].iloc[0]:
        st.caption(row["Name"].iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo live", f"{price:.2f}" if price is not None else "—")
    m2.metric("Graham Number", f"{gn:.2f}" if gn else "—")
    m3.metric("Margine di sicurezza", f"{mos*100:.1f}%" + (" " + mos_star(mos) if mos is not None else "") if mos is not None else "—")

    with st.expander("Dettagli formula GN"):
        st.markdown("""
**GN = √(22.5 × EPS × BVPS)**  
Dove *EPS* = utili per azione, *BVPS* = valore contabile per azione.  
Se EPS o BVPS ≤ 0 il GN non è definito.
                """.strip())

    # Link esterni se presenti
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**ISIN**: {row['ISIN'].iloc[0] if not row.empty else '—'}")
    with c2:
        inv = row["InvestingURL"].iloc[0] if not row.empty else ""
        if inv: st.markdown(f"[Scheda Investing]({inv})")
    with c3:
        ms = row["MorningstarURL"].iloc[0] if not row.empty else ""
        if ms: st.markdown(f"[Scheda Morningstar]({ms})")

    # Snapshot singolo
    st.divider()
    st.subheader("Snapshot su Storico")
    if can_write_actions():
        if st.button("➕ Salva snapshot TICKER"):
            _, _, _, hist_ws = get_client_and_ws()
            tz = pytz.timezone("Europe/Rome")
            now = dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            eps_v = eps if eps is not None else ""
            bvps_v = bvps if bvps is not None else ""
            gn_v = gn if gn is not None else ""
            delta = (gn - price) if (gn and price is not None) else ""
            mos_v = ((gn - price) / gn) if (gn and price is not None) else ""
            row_out = [now, ticker, price or "", eps_v, bvps_v, gn_v, "Yahoo Finance", price or "", gn_v, delta, mos_v]
            append_history(hist_ws, row_out)
            st.success("Snapshot salvato ✅")
    else:
        st.info("Inserisci l'admin code per abilitare lo snapshot (scrittura).")

# Tabella riassuntiva (opzionale)
with st.expander("Vedi tabella Fondamentali (read-only)"):
    df_view = df.copy()
    # MOS su ultimo prezzo live è costoso per tutti i ticker: lo evitiamo qui (solo read)
    st.dataframe(df_view, use_container_width=True)
