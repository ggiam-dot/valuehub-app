import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo
import json, re
import time

import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG (da Secrets)
# =========================
SHEET_ID  = st.secrets.get("google_sheet_id")                  # obbligatorio
FUND_TAB  = st.secrets.get("fund_tab", "Fondamentali")
HIST_TAB  = st.secrets.get("hist_tab", "Storico")
YF_SUFFIX = st.secrets.get("yf_suffix", ".MI")

# Colonne per lettera (MAIUSCOLE). Default: A=Ticker, B=EPS, C=BVPS, D=Graham
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")
# Opzionale: colonna Nome. Se vuota, useremo Yahoo come fallback.
NAME_LETTER   = st.secrets.get("name_col_letter", "")

st.set_page_config(page_title="Value Hub – Graham Lookup", page_icon="📈", layout="centered")
st.title("📈 Value Investment – Graham Lookup")

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
    s = str(x).strip().replace("\u00A0","")
    s = s.replace("\u2212","-").replace("€","").replace("EUR","").replace("’","'")
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
        if "," in s: s = s.replace(",", ".")
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

@st.cache_data(show_spinner=False)
def fetch_company_name_yf(symbol: str) -> str:
    """Nome società da Yahoo (cache)."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        name = info.get("shortName") or info.get("longName") or ""
        return str(name)
    except Exception:
        return ""

def gn_formula_225(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
    return sqrt(22.5 * eps * bvps)

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    row = [ts, ticker, price if price is not None else "", eps, bvps, graham if graham is not None else "", fonte]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def load_history():
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty:
        return dfh
    if "Timestamp" in dfh.columns:
        dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"], errors="coerce")
    return dfh

def last_eod_for_ticker(ticker: str):
    dfh = load_history()
    if dfh.empty or "Ticker" not in dfh.columns or "Timestamp" not in dfh.columns: return None
    dft = dfh[dfh["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if dft.empty: return None
    dft = dft.sort_values("Timestamp")
    return dft.iloc[-1].to_dict()

def has_global_eod_today(today_date: date) -> bool:
    """True se esiste almeno un record con data odierna (per evitare doppi EOD)."""
    dfh = load_history()
    if dfh.empty or "Timestamp" not in dfh.columns: return False
    days = pd.to_datetime(dfh["Timestamp"], errors="coerce").dt.date
    return any(days == today_date)

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
    idx_name   = _letter_to_index(NAME_LETTER) if NAME_LETTER else -1

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

    if 0 <= idx_name < ncols:
        df["Name_raw"] = [row[idx_name] if idx_name < len(row) else "" for row in data]
    else:
        df["Name_raw"] = ""

    mask_nonempty = (df["Ticker_raw"].astype(str).str.strip()!="") | \
                    (df["EPS_raw"].astype(str).str.strip()!="")   | \
                    (df["BVPS_raw"].astype(str).str.strip()!="")
    df = df[mask_nonempty].reset_index(drop=True)

    df["Ticker"]   = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["EPS"]      = df["EPS_raw"].apply(to_number)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number)
    df["Name"]     = df["Name_raw"].astype(str).str.strip()

    meta = {
        "ticker_letter": TICKER_LETTER,
        "eps_letter": EPS_LETTER,
        "bvps_letter": BVPS_LETTER,
        "gn_letter": GN_LETTER,
        "name_letter": NAME_LETTER,
        "header_row": header
    }
    return df, meta

# =========================
# GN helpers / EOD helpers
# =========================
def compute_gn_series(df):
    out = []
    for e, b in zip(df["EPS"], df["BVPS"]):
        if e is None or b is None or e <= 0 or b <= 0:
            out.append("")
        else:
            out.append((22.5 * e * b) ** 0.5)
    return pd.Series(out)

def write_gn_to_sheet(ws_fund, gn_series, gn_letter="D"):
    if gn_series is None or len(gn_series) == 0:
        return
    start_row = 2
    end_row   = start_row + len(gn_series) - 1
    cell_range = f'{gn_letter}{start_row}:{gn_letter}{end_row}'
    out = [[("" if (v is None or v == "") else float(v))] for v in gn_series]
    ws_fund.update(cell_range, out, value_input_option="USER_ENTERED")

def snapshot_all_tickers(df):
    total = len(df)
    if total == 0:
        return 0
    prog = st.progress(0, text="Snapshot EOD in corso…")
    done = 0
    now_str = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M:%S")
    for _, r in df.iterrows():
        tick = r["Ticker"]
        if not tick:
            done += 1; prog.progress(done/total); continue
        price = fetch_price_yf(normalize_symbol(tick))
        append_history_row(now_str, tick, price, r["EPS"], r["BVPS"], r["GN_sheet"], "Auto EOD")
        done += 1
        prog.progress(done/total)
        time.sleep(0.02)
    return done

# =========================
# UI
# =========================
df, meta = load_fundamentals_by_letter()
if df.empty or df["Ticker"].isna().all():
    st.warning("Nessun dato utile. Controlla che il foglio contenga Ticker(A), EPS(B), BVPS(C), Graham(D).")
else:
    # ---- Ricerca Ticker/Nome ----
    st.text_input("🔎 Cerca (Ticker o Nome)…", key="search_query", placeholder="es. ENEL.MI o Enel")
    q = (st.session_state.get("search_query") or "").strip().lower()

    def get_display_name(tick):
        name_sheet = df.loc[df["Ticker"] == tick, "Name"].fillna("").astype(str).str.strip()
        if not name_sheet.empty and name_sheet.iloc[0]:
            return name_sheet.iloc[0]
        return fetch_company_name_yf(normalize_symbol(tick)) or ""

    tickers_all = df["Ticker"].replace("", np.nan).dropna().tolist()
    if q:
        tickers = [t for t in tickers_all if (q in t.lower()) or (q in get_display_name(t).lower())]
    else:
        tickers = tickers_all

    tick = st.selectbox("Scegli il Ticker", options=tickers)

    if tick:
        row = df[df["Ticker"] == tick].iloc[0].to_dict()
        eps_val    = row.get("EPS")
        bvps_val   = row.get("BVPS")
        gn_sheet   = row.get("GN_sheet")
        gn_formula = gn_formula_225(eps_val, bvps_val)

        symbol = normalize_symbol(tick)
        company_name = get_display_name(tick)
        st.markdown(f"### {tick} — {company_name}")

        price_live = fetch_price_yf(symbol)

        margin_pct = None
        if price_live is not None and gn_sheet is not None and gn_sheet > 0:
            margin_pct = (1 - (price_live / gn_sheet)) * 100

        # ultimo snapshot: mostra solo se esiste (niente messaggio "nessuno snapshot")
        eod = last_eod_for_ticker(tick)
        if eod and eod.get("Timestamp"):
            ts = pd.to_datetime(eod["Timestamp"])
            st.success(f"✅ Ultimo snapshot: {ts.strftime('%Y-%m-%d %H:%M:%S')}")

        # metriche
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

        # --- TAB: Dettaglio / Storico ---
        tab1, tab2 = st.tabs(["Dettaglio", "Storico"])

        with tab1:
            # Formula
            st.markdown("### Formula")
            if gn_formula is not None:
                st.code(f"√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_formula:.4f}")
            else:
                st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

            st.markdown("---")
            # Pulsanti affiancati, stessa dimensione
            bcol1, bcol2, bcol3 = st.columns([1,1,2])
            with bcol1:
                if st.button("🔄 Aggiorna dal foglio"):
                    st.cache_data.clear()
                    st.rerun()
            with bcol2:
                if st.button("✍️ Riscrivi Graham# (22,5)"):
                    try:
                        gn_series = compute_gn_series(df)
                        write_gn_to_sheet(ws_fund, gn_series, gn_letter=GN_LETTER)
                        st.success("Colonna Graham# riscritta (22,5×EPS×BVPS).")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore durante la riscrittura: {e}")
            with bcol3:
                if st.button("💾 Salva snapshot su 'Storico'"):
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_history_row(now_str, tick, price_live, eps_val, bvps_val, gn_sheet, "App (GN da Sheet)")
                    st.success("Snapshot salvato su 'Storico'.")

        with tab2:
            dfh = load_history()
            if not dfh.empty and "Ticker" in dfh.columns:
                dft = dfh[dfh["Ticker"].astype(str).str.upper() == tick.upper()].copy()
                if not dft.empty:
                    dft = dft.sort_values("Timestamp")
                    min_day = pd.to_datetime(dft["Timestamp"]).dt.date.min()
                    max_day = pd.to_datetime(dft["Timestamp"]).dt.date.max()
                    start, end = st.date_input("Intervallo date",
                                               value=(min_day, max_day),
                                               min_value=min_day, max_value=max_day)
                    if isinstance(start, date) and isinstance(end, date) and start <= end:
                        dft = dft[(pd.to_datetime(dft["Timestamp"]).dt.date >= start) &
                                  (pd.to_datetime(dft["Timestamp"]).dt.date <= end)]

                    show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Fonte"] if c in dft.columns]
                    st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

                    csv = dft[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Scarica CSV", data=csv, file_name=f"storico_{tick}.csv", mime="text/csv")

                    if "Price" in dft.columns and "Graham" in dft.columns:
                        plot_df = dft[["Timestamp","Price","Graham"]].dropna()
                        plot_df = plot_df.set_index("Timestamp")
                        st.line_chart(plot_df, use_container_width=True)

        # DEBUG in fondo pagina
        st.markdown("---")
        with st.expander("🔎 Debug colonne & valori"):
            st.write(pd.DataFrame({
                "Campo": ["Ticker_letter","EPS_letter","BVPS_letter","GN_letter","Name_letter",
                          "Ticker_raw","EPS_raw","BVPS_raw","GN_sheet_raw","Name_raw",
                          "EPS_parsed","BVPS_parsed","GN_sheet_parsed","GN_formula_22_5","Nome_finale"],
                "Valore": [
                    meta.get("ticker_letter"), meta.get("eps_letter"), meta.get("bvps_letter"), meta.get("gn_letter"), meta.get("name_letter"),
                    row.get("Ticker_raw"), row.get("EPS_raw"), row.get("BVPS_raw"), row.get("GN_sheet_raw"), row.get("Name_raw",""),
                    eps_val, bvps_val, gn_sheet, gn_formula, company_name
                ]
            }))

    # =========================
    # AUTO-SNAPSHOT EOD (TUTTI I TITOLI)
    # =========================
    now_rome = datetime.now(ZoneInfo("Europe/Rome"))
    if now_rome.time() >= dtime(hour=17, minute=30):
        if not has_global_eod_today(now_rome.date()):
            try:
                count = snapshot_all_tickers(df)
                st.success(f"📌 Snapshot EOD globale creato: {count} titoli.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Errore EOD automatico: {e}")
