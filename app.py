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

st.set_page_config(page_title="Value Hub – Graham Lookup", page_icon="📈", layout="centered")
st.title("📈 Value Investment – Graham Lookup")
st.caption("GN letto dallo Sheet (colonne per lettera). Prezzo live via Yahoo. Formula mostrata (22.5×EPS×BVPS).")

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
    if not letter: return -1
    s = letter.strip().upper()
    n = 0
    for ch in s:
        if not ("A" <= ch <= "Z"): return -1
        n = n*26 + (ord(ch)-64)
    return n-1  # zero-based

def to_number(x):
    """Parser robusto: IT/US, €, %, migliaia, virgole/punti."""
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace("\u00A0","")    # NBSP
    s = s.replace("\u2212","-")                # minus unicode
    s = s.replace("€","").replace("EUR","").replace("’","'")
    has_pct = "%" in s
    s = s.replace("%","")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if s in {"", "-", ","}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".","").replace(",",".")   # IT 1.234,56 -> 1234.56
        else:
            s = s.replace(",","")                    # US 1,234.56 -> 1234.56
    else:
        if "," in s:
            s = s.replace(",", ".")
        elif s.count(".") > 1:
            parts = s.split("."); s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        v = float(s)
        return v/100.0 if has_pct else v
    except:
        return None

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
            if not h.empty: price = float(h["Close"].dropna().iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None

def gn_formula_225(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
    return sqrt(22.5 * eps * bvps)

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    row = [ts, ticker, price if price is not None else "", eps, bvps, graham if graham is not None else "", fonte]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def last_eod_for_ticker(ticker: str):
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty or "Ticker" not in dfh.columns or "Timestamp" not in dfh.columns: return None
    dft = dfh[dfh["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if dft.empty: return None
    dft["Timestamp"] = pd.to_datetime(dft["Timestamp"], errors="coerce")
    dft = dft.sort_values("Timestamp")
    return dft.iloc[-1].to_dict()

# =========================
# LOAD DATA (per lettera)
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(), {}

    header = values[0]
    data = values[1:]
    ncols = len(header)

    idx_ticker = _letter_to_index(TICKER_LETTER)
    idx_eps    = _letter_to_index(EPS_LETTER)
    idx_bvps   = _letter_to_index(BVPS_LETTER)
    idx_gn     = _letter_to_index(GN_LETTER)

    for name, idx in [("Ticker",idx_ticker), ("EPS",idx_eps), ("BVPS",idx_bvps), ("GN",idx_gn)]:
        if idx < 0 or idx >= ncols:
            st.error(f"Lettera colonna {name} non valida o fuori range.")
            return pd.DataFrame(), {}

    df = pd.DataFrame({
        "Ticker_raw":  [row[idx_ticker] if idx_ticker < len(row) else "" for row in data],
        "EPS_raw":     [row[idx_eps]    if idx_eps    < len(row) else "" for row in data],
        "BVPS_raw":    [row[idx_bvps]   if idx_bvps   < len(row) else "" for row in data],
        "GN_sheet_raw":[row[idx_gn]     if idx_gn     < len(row) else "" for row in data],
    })

    # togli righe completamente vuote
    mask_nonempty = (df["Ticker_raw"].astype(str).str.strip()!="") | \
                    (df["EPS_raw"].astype(str).str.strip()!="")   | \
                    (df["BVPS_raw"].astype(str).str.strip()!="")
    df = df[mask_nonempty].reset_index(drop=True)

    # normalizza
    df["Ticker"]   = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["EPS"]      = df["EPS_raw"].apply(to_number)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number)

    meta = {
        "ticker_letter": TICKER_LETTER,
        "eps_letter": EPS_LETTER,
        "bvps_letter": BVPS_LETTER,
        "gn_letter": GN_LETTER,
        "header_row": header
    }
    return df, meta

# =========================
# BUTTON: refresh
# =========================
col_btn, _ = st.columns([1,3])
with col_btn:
    if st.button("🔄 Aggiorna dal foglio"):
        st.cache_data.clear()
        st.rerun()   # versione nuova (niente experimental)

# =========================
# FUNZIONI: riscrivere GN su Sheet
# =========================
def compute_gn_series(df):
    """Series con GN = sqrt(22.5 * EPS * BVPS); '' se non calcolabile."""
    out = []
    for e, b in zip(df["EPS"], df["BVPS"]):
        if e is None or b is None or e <= 0 or b <= 0:
            out.append("")
        else:
            out.append((22.5 * e * b) ** 0.5)
    return pd.Series(out)

def write_gn_to_sheet(ws_fund, gn_series, gn_letter="D"):
    """Scrive GN in colonna gn_letter, dalla riga 2 in giù (sovrascrive valori/ formule esistenti)."""
    if gn_series is None or len(gn_series) == 0:
        return
    start_row = 2
    end_row   = start_row + len(gn_series) - 1
    cell_range = f'{gn_letter}{start_row}:{gn_letter}{end_row}'
    out = [[("" if (v is None or v == "") else float(v))] for v in gn_series]
    ws_fund.update(cell_range, out, value_input_option="USER_ENTERED")

# =========================
# UI
# =========================
df, meta = load_fundamentals_by_letter()
if df.empty or df["Ticker"].isna().all():
    st.warning("Nessun dato utile. Controlla che il foglio contenga Ticker(A), EPS(B), BVPS(C), Graham(D).")
else:
    tickers = df["Ticker"].replace("", np.nan).dropna().tolist()
    tick = st.selectbox("Scegli il Ticker", options=tickers)

    if tick:
        row = df[df["Ticker"] == tick].iloc[0].to_dict()
        eps_val    = row.get("EPS")
        bvps_val   = row.get("BVPS")
        gn_sheet   = row.get("GN_sheet")               # GN letto dal foglio
        gn_formula = gn_formula_225(eps_val, bvps_val) # formula mostrata (controllo)

        # Prezzo live
        symbol = normalize_symbol(tick)
        price_live = fetch_price_yf(symbol)

        # Margine % vs GN_sheet
        margin_pct = None
        if price_live is not None and gn_sheet is not None and gn_sheet > 0:
            margin_pct = (1 - (price_live / gn_sheet)) * 100

        # Snap EOD
        eod = last_eod_for_ticker(tick)
        if eod and eod.get("Timestamp"):
            ts = pd.to_datetime(eod["Timestamp"])
            st.success(f"✅ Ultimo snapshot: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("ℹ️ Nessuno snapshot EOD trovato per questo ticker.")

        # Metriche
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
            st.caption(f"Fonte: Yahoo – {symbol}")
        with c2:
            st.metric("Numero di Graham (da sheet)", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
            st.caption(f"Lettera GN: {meta.get('gn_letter')}")
        with c3:
            if margin_pct is not None:
                label = "✅ Sottovalutata" if margin_pct > 0 else "❌ Sopravvalutata"
                st.metric("Margine di sicurezza", f"{margin_pct:.2f}%", label)
            else:
                st.metric("Margine di sicurezza", "n/d")

        # Formula informativa (per confronto)
        st.markdown("### Formula (mostrata; GN usato = valore dello Sheet)")
        if gn_formula is not None:
            st.code(f"√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_formula:.4f}")
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        # Debug
        with st.expander("🔎 Debug colonne & valori"):
            st.write(pd.DataFrame({
                "Campo": ["Ticker_letter","EPS_letter","BVPS_letter","GN_letter",
                          "Ticker_raw","EPS_raw","BVPS_raw","GN_sheet_raw",
                          "EPS_parsed","BVPS_parsed","GN_sheet_parsed","GN_formula_22_5"],
                "Valore": [
                    meta.get("ticker_letter"), meta.get("eps_letter"), meta.get("bvps_letter"), meta.get("gn_letter"),
                    row.get("Ticker_raw"), row.get("EPS_raw"), row.get("BVPS_raw"), row.get("GN_sheet_raw"),
                    eps_val, bvps_val, gn_sheet, gn_formula
                ]
            }))

        st.markdown("---")
        # Bottone: riscrivi GN corretto in colonna D
        if st.button("✍️ Riscrivi Graham# su Sheet (22,5)"):
            try:
                gn_series = compute_gn_series(df)
                write_gn_to_sheet(ws_fund, gn_series, gn_letter=GN_LETTER)
                st.success("Colonna Graham# riscritta con i valori corretti (22,5×EPS×BVPS).")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Errore durante la riscrittura: {e}")

        # Salva snapshot (usa GN da sheet)
        if st.button("💾 Salva snapshot su 'Storico'"):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_history_row(now_str, tick, price_live, eps_val, bvps_val, gn_sheet, "App (GN da Sheet)")
            st.success("Snapshot salvato su 'Storico'.")
