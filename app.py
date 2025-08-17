import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
from datetime import datetime, date
from zoneinfo import ZoneInfo
import json, re, time

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
# Opzionale: colonna Nome (se non presente, useremo Yahoo)
NAME_LETTER   = st.secrets.get("name_col_letter", "")

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup",
                   page_icon="📈", layout="centered")
st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")

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
    s = str(x).strip().replace("\u00A0","")
    s = s.replace("€","").replace("EUR","").replace("%","").replace("\u2212","-")
    s = re.sub(r"[^0-9\-,\.]", "", s)
    if s in {"", "-", ","}: return None
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
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return str(info.get("shortName") or info.get("longName") or "")
    except Exception:
        return ""

def gn_formula_225(eps, bvps):
    if eps is None or bvps is None or eps <= 0 or bvps <= 0: return None
    return sqrt(22.5 * eps * bvps)

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

def append_history_row(ts, ticker, price, eps, bvps, graham, fonte="App"):
    ensure_history_headers()
    delta = margin = None
    if graham not in (None, "", 0) and price not in (None, ""):
        delta  = float(price) - float(graham)
        margin = (1 - (float(price)/float(graham))) * 100
    row = [
        ts, ticker,
        ("" if price is None else float(price)),
        ("" if eps is None else float(eps)),
        ("" if bvps is None else float(bvps)),
        ("" if graham is None else float(graham)),
        ("" if delta is None else float(delta)),
        ("" if margin is None else float(margin)),
        fonte
    ]
    ws_hist.append_row(row, value_input_option="USER_ENTERED")

@st.cache_data(show_spinner=False)
def load_history():
    ensure_history_headers()
    recs = ws_hist.get_all_records()
    dfh = pd.DataFrame(recs)
    if dfh.empty: return dfh
    if "Timestamp" in dfh.columns:
        dfh["Timestamp"] = pd.to_datetime(dfh["Timestamp"], errors="coerce")
    for col in ["Delta","MarginPct"]:
        if col not in dfh.columns: dfh[col] = np.nan
    if not dfh.empty:
        _p = pd.to_numeric(dfh.get("Price"), errors="coerce")
        _g = pd.to_numeric(dfh.get("Graham"), errors="coerce")
        mask = dfh["Delta"].isna() | dfh["MarginPct"].isna()
        dfh.loc[mask, "Delta"] = (_p - _g).where((_p.notna()) & (_g.notna()))
        dfh.loc[mask, "MarginPct"] = (1 - (_p/_g))*100
    return dfh

# =========================
# LOAD FUNDAMENTALS (per lettera)
# =========================
@st.cache_data(show_spinner=False)
def load_fundamentals_by_letter():
    values = ws_fund.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(), {}
    header, data = values[0], values[1:]
    ncols = len(header)

    idx_ticker = _letter_to_index(TICKER_LETTER)
    idx_eps    = _letter_to_index(EPS_LETTER)
    idx_bvps   = _letter_to_index(BVPS_LETTER)
    idx_gn     = _letter_to_index(GN_LETTER)
    idx_name   = _letter_to_index(NAME_LETTER) if NAME_LETTER else -1

    df = pd.DataFrame({
        "Ticker_raw":  [row[idx_ticker] if idx_ticker < len(row) else "" for row in data],
        "EPS_raw":     [row[idx_eps]    if idx_eps    < len(row) else "" for row in data],
        "BVPS_raw":    [row[idx_bvps]   if idx_bvps   < len(row) else "" for row in data],
        "GN_sheet_raw":[row[idx_gn]     if idx_gn     < len(row) else "" for row in data],
    })
    df["Name_raw"] = [row[idx_name] if (0 <= idx_name < len(row)) else "" for row in data] if idx_name >= 0 else ""

    df = df[(df["Ticker_raw"].astype(str).str.strip()!="")].reset_index(drop=True)
    df["Ticker"]   = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["EPS"]      = df["EPS_raw"].apply(to_number)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number)
    df["Name"]     = (df["Name_raw"].astype(str).str.strip() if isinstance(df["Name_raw"], pd.Series) else "")

    return df, {"name_from_sheet": (idx_name >= 0)}

# =========================
# UI
# =========================
df, meta = load_fundamentals_by_letter()
if df.empty:
    st.warning("Nessun dato utile. Controlla il foglio.")
else:
    # ------ label "TICKER — Nome" per abilitare ricerca anche per nome ------
    @st.cache_data(show_spinner=False)
    def get_display_name(tick: str) -> str:
        r = df[df["Ticker"] == tick]
        if not r.empty:
            n = str(r.iloc[0].get("Name") or "").strip()
            if n:
                return n
        return fetch_company_name_yf(normalize_symbol(tick)) or ""

    tickers_all = sorted(df["Ticker"].tolist())
    labels = []
    label_to_ticker = {}
    for t in tickers_all:
        nm = get_display_name(t)
        lab = f"{t} — {nm}" if nm else t
        labels.append(lab)
        label_to_ticker[lab] = t

    tab1, tab2 = st.tabs(["📊 Analisi", "📜 Storico"])

    with tab1:
        selected_label = st.selectbox("Scegli il Ticker", options=labels, index=0)
        tick = label_to_ticker[selected_label]

        row = df[df["Ticker"]==tick].iloc[0]
        eps_val, bvps_val, gn_sheet = row["EPS"], row["BVPS"], row["GN_sheet"]
        gn_formula = gn_formula_225(eps_val, bvps_val)
        symbol = normalize_symbol(tick)
        price_live = fetch_price_yf(symbol)

        # HEADER con icone accanto al nome
        company_name = get_display_name(tick)
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
              <h3 style="margin:0">{tick} — {company_name}</h3>
              <span style="display:inline-flex;gap:8px;align-items:center;">
                <a href="https://finance.yahoo.com/quote/{tick}" target="_blank" rel="noopener" title="Yahoo Finance">
                  <img src="https://www.google.com/s2/favicons?sz=32&domain=finance.yahoo.com" style="width:16px;height:16px;">
                </a>
                <a href="https://www.google.com/search?q=Investing+{tick}" target="_blank" rel="noopener" title="Investing">
                  <img src="https://www.google.com/s2/favicons?sz=32&domain=it.investing.com" style="width:16px;height:16px;">
                </a>
                <a href="https://www.google.com/search?q=Morningstar+{tick}" target="_blank" rel="noopener" title="Morningstar">
                  <img src="https://www.google.com/s2/favicons?sz=32&domain=morningstar.com" style="width:16px;height:16px;">
                </a>
              </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================
        # Metriche principali + badge valutazione
        # =========================
        margin_pct = (1 - (price_live/gn_sheet))*100 if (price_live is not None and gn_sheet not in (None,0)) else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
        with c2:
            st.metric("Graham#", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
        with c3:
            if margin_pct is not None:
                # Percentuale grande + etichetta sotto (minimal)
                pct = f"{margin_pct:.2f}%"
                if margin_pct > 33:
                    # G dorata minimal (colore gold)
                    html = f"""
                    <div style="text-align:center;">
                      <div style="font-weight:800; font-size:20px; color:#0a7f2e;">{pct}</div>
                      <div style="margin-top:4px; font-weight:700; color:#0a7f2e;">
                        <span style="color:#DAA520;">G</span>
                      </div>
                    </div>"""
                elif margin_pct > 0:
                    html = f"""
                    <div style="text-align:center;">
                      <div style="font-weight:800; font-size:20px; color:#0a7f2e;">{pct}</div>
                      <div style="margin-top:4px; font-weight:700; color:#0a7f2e;">Sottovalutata</div>
                    </div>"""
                else:
                    html = f"""
                    <div style="text-align:center;">
                      <div style="font-weight:800; font-size:20px; color:#b00020;">{pct}</div>
                      <div style="margin-top:4px; font-weight:700; color:#b00020;">Sopravvalutata</div>
                    </div>"""
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.metric("Margine", "n/d")

        # =========================
        # Formula (più grande ed evidente)
        # =========================
        st.markdown("### The GN Formula")
        if gn_formula is not None:
            st.markdown(
                f"<div style='font-size:18px; font-weight:700; color:#222;'>√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_formula:.4f}</div>",
                unsafe_allow_html=True
            )
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        st.markdown("---")
        # ===== Toggle Admin (sopra i 3 bottoni) =====
        is_admin = st.toggle("🛠️ Modalità amministratore", value=True,
                             help="Mostra/nasconde i comandi di amministrazione")

        if is_admin:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Aggiorna dal foglio", use_container_width=True):
                    st.cache_data.clear(); st.rerun()
            with col2:
                if st.button("✍️ Riscrivi Graham# su Sheet", use_container_width=True):
                    gn_series = df.apply(lambda r: gn_formula_225(r["EPS"], r["BVPS"]), axis=1)
                    out = [[("" if (v is None or pd.isna(v)) else float(v))] for v in gn_series]
                    start_row = 2
                    end_row = start_row + len(out) - 1
                    ws_fund.update(f"{GN_LETTER}{start_row}:{GN_LETTER}{end_row}", out,
                                   value_input_option="USER_ENTERED")
                    st.success("Colonna Graham# aggiornata (22,5×EPS×BVPS).")
                    st.cache_data.clear(); st.rerun()
            with col3:
                if st.button("💾 Salva snapshot", use_container_width=True):
                    now_str = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d %H:%M:%S")
                    append_history_row(now_str, tick, price_live, eps_val, bvps_val, gn_sheet, "App (GN da Sheet)")
                    st.success("Snapshot salvato nello Storico.")

    with tab2:
        dfh = load_history()
        if not dfh.empty:
            try:
                current_tick = label_to_ticker[selected_label]
            except Exception:
                current_tick = None

            dft = dfh[dfh["Ticker"].astype(str).str.upper() == (current_tick or "").upper()] if current_tick else dfh
            dft = dft.sort_values("Timestamp")

            # 🔔 Ultimo snapshot SOLO QUI, sopra Intervallo date
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

                show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Delta","MarginPct","Fonte"] if c in dft.columns]
                st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

                csv = dft[show_cols].to_csv(index=False).encode("utf-8")
                fname = f"storico_{(current_tick or 'ALL')}.csv"
                st.download_button("⬇️ Scarica CSV", data=csv, file_name=fname, mime="text/csv")

                if ("Price" in dft.columns) and ("Graham" in dft.columns):
                    plot_df = dft[["Timestamp","Price","Graham"]].dropna().set_index("Timestamp")
                    st.line_chart(plot_df, use_container_width=True)

        # Debug in fondo (solo admin, se definito in tab1)
        if 'is_admin' in locals() and is_admin:
            st.markdown("---")
            with st.expander("🔎 Debug"):
                st.write(df.head())
