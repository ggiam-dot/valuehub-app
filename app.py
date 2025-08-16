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
SHEET_ID  = st.secrets.get("google_sheet_id")                  # obbligatorio
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

# Lettere di colonna (MAIUSCOLE). Default: A=Ticker, B=EPS, C=BVPS, D=Graham
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")

st.set_page_config(page_title="Vigil – Value_Investment_Graham_Intelligent_Lookup",
                   page_icon="📈", layout="centered")
st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")

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
    s = (letter or "").strip().upper()
    n = 0
    for ch in s:
        n = n*26 + (ord(ch)-64)
    return n-1

def to_number(x):
    if x is None: return None
    if isinstance(x, (int,float)): return float(x)
    s = str(x).strip().replace("\u00A0","")
    s = s.replace("€","").replace("EUR","").replace("%","").replace("\u2212","-")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".","").replace(",",".")   # IT 1.234,56 -> 1234.56
        else:
            s = s.replace(",","")                    # US 1,234.56 -> 1234.56
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s if "." in s else s + YF_SUFFIX

@st.cache_data(show_spinner=False)
def fetch_price_yf(symbol: str):
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        price = fi.get("last_price") if fi else None
        if price is None:
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
        if price is None:
            h = t.history(period="5d")
            if not h.empty: price = float(h["Close"].dropna().iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

def gn_formula_225(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
    return sqrt(22.5 * eps * bvps)

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    # calcola Delta e MarginPct
    delta = None
    margin = None
    if graham not in (None, "", 0) and price not in (None, ""):
        delta = float(price) - float(graham)
        margin = (1 - (float(price)/float(graham))) * 100
    row = [ts, ticker,
           ("" if price is None else float(price)),
           ("" if eps is None else float(eps)),
           ("" if bvps is None else float(bvps)),
           ("" if graham is None else float(graham)),
           ("" if delta is None else float(delta)),
           ("" if margin is None else float(margin)),
           fonte]
    # Assicura header completo
    ensure_history_headers()
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

# ---------- Storico: header + delta/margine ----------
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
        ws_hist.insert_row(DESIRED_HIST_HEADER, 1)
        return
    header = values[0]; changed = False
    for col in DESIRED_HIST_HEADER:
        if col not in header:
            header.append(col); changed = True
    if changed:
        ws_hist.update(f"A1:{_col_letter(len(header))}1", [header], value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def load_history():
    ensure_history_headers()
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty: return dfh
    if "Timestamp" in dfh.columns:
        dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"], errors="coerce")
    # garantisci colonne Delta/Margin anche se righe vecchie
    if "Delta" not in dfh.columns: dfh["Delta"] = np.nan
    if "MarginPct" not in dfh.columns: dfh["MarginPct"] = np.nan
    if not dfh.empty:
        _p = pd.to_numeric(dfh.get("Price"), errors="coerce")
        _g = pd.to_numeric(dfh.get("Graham"), errors="coerce")
        mask = dfh["Delta"].isna() | dfh["MarginPct"].isna()
        dfh.loc[mask, "Delta"] = (_p - _g).where((_p.notna()) & (_g.notna()))
        dfh.loc[mask, "MarginPct"] = (1 - (_p/_g))*100
    return dfh

# =========================
# LOAD FUNDAMENTALS
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    header, data = values[0], values[1:]
    idx_t = _letter_to_index(TICKER_LETTER)
    idx_e = _letter_to_index(EPS_LETTER)
    idx_b = _letter_to_index(BVPS_LETTER)
    idx_g = _letter_to_index(GN_LETTER)
    df = pd.DataFrame({
        "Ticker":[(row[idx_t] if idx_t < len(row) else "") for row in data],
        "EPS":[to_number(row[idx_e]) if idx_e < len(row) else None for row in data],
        "BVPS":[to_number(row[idx_b]) if idx_b < len(row) else None for row in data],
        "GN_sheet":[to_number(row[idx_g]) if idx_g < len(row) else None for row in data],
    })
    df = df[df["Ticker"].astype(str).str.strip()!=""].reset_index(drop=True)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df

df = load_fundamentals_by_letter()
tickers = sorted(df["Ticker"].dropna().unique().tolist())

# =========================
# UI – Tabs
# =========================
tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

with tab1:
    tick = st.selectbox("Scegli il Ticker", tickers)
    if tick:
        row = df[df["Ticker"]==tick].iloc[0]
        eps, bvps, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_calc = gn_formula_225(eps, bvps)
        price = fetch_price_yf(normalize_symbol(tick))

        margin_pct = (1 - (price/gn_sheet))*100 if (price is not None and gn_sheet not in (None,0)) else None

        c1,c2,c3 = st.columns(3)
        c1.metric("Prezzo live", f"{price:.2f}" if price is not None else "n/d")
        c2.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
        c3.metric("Margine", f"{margin_pct:.2f}%" if margin_pct is not None else "n/d")

        st.markdown("### The GN Formula")
        if gn_calc is not None:
            st.code(f"√(22.5 × {eps:.4f} × {bvps:.4f}) = {gn_calc:.4f}")
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        st.markdown("---")
        # ===== Toggle Admin QUI, subito sopra i pulsanti =====
        current_admin = st.session_state.get("is_admin", True)
        is_admin = st.toggle("🛠️ Modalità amministratore", value=current_admin,
                             help="Mostra/nasconde i comandi di amministrazione",
                             key="admin_toggle")
        st.session_state["is_admin"] = is_admin

        # Pulsanti (visibili solo se Admin attivo)
        if is_admin:
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔄 Aggiorna dal foglio"):
                    st.cache_data.clear(); st.rerun()
            with b2:
                if st.button("✍️ Riscrivi Graham# su Sheet"):
                    # ricalcola serie GN e scrivi in blocco
                    gn_series = df.apply(lambda r: gn_formula_225(r["EPS"], r["BVPS"]), axis=1)
                    out = [[("" if pd.isna(v) else float(v))] for v in gn_series]
                    start_row = 2
                    end_row = start_row + len(out) - 1
                    ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}", out, value_input_option="USER_ENTERED")
                    st.success("Colonna Graham# aggiornata (22,5×EPS×BVPS).")
                    st.cache_data.clear(); st.rerun()

with tab2:
    dfh = load_history()
    if not dfh.empty:
        # Filtra per ticker scelto (selezionato in tab1)
        dft = dfh[dfh["Ticker"].astype(str).str.upper() == (tick or "").upper()].copy() if tick else dfh.copy()
        dft = dft.sort_values("Timestamp")

        # 🔔 Ultimo snapshot SOLO QUI, subito sopra Intervallo date
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

            # mostra con Delta/MarginPct se presenti
            show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"] if c in dft.columns]
            st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

            csv = dft[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Scarica CSV", data=csv, file_name=f"storico_{(tick or 'ALL')}.csv", mime="text/csv")

# Debug in fondo pagina (solo Admin)
if st.session_state.get("is_admin", True):
    st.markdown("---")
    with st.expander("🔎 Debug"):
        st.write(df.head())
