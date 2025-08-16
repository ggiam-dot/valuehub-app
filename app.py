import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt
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

# Colonne base per LETTERA (MAIUSCOLE). Default: A=Ticker, B=EPS, C=BVPS, D=Graham
TICKER_LETTER = st.secrets.get("ticker_col_letter", "A")
EPS_LETTER    = st.secrets.get("eps_col_letter", "B")
BVPS_LETTER   = st.secrets.get("bvps_col_letter", "C")
GN_LETTER     = st.secrets.get("gn_col_letter", "D")
# OPZIONALE: colonna Nome (se presente). Se assente, useremo Yahoo come fallback.
NAME_LETTER   = st.secrets.get("name_col_letter", "")

st.set_page_config(page_title="Vigil-(Value_Investment_Graham_Intelligent_Lookup)", page_icon="📈", layout="centered")

# --- stile minimal-elegante (mobile friendly) ---
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
h1 { font-size: 1.6rem; }
h3 { margin-top: 0.5rem; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #eee; border-radius: 16px; padding: 10px 12px; }
div[data-testid="stMetricLabel"] { color: #6b7280; font-weight: 500; }
div[data-testid="stMetricValue"] { font-weight: 700; }
input[type="text"] { border-radius: 12px !important; }
.badge-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:6px; }
.badge { display:inline-flex; gap:6px; align-items:center; border:1px solid #eee; border-radius:999px; padding:6px 10px; text-decoration:none; color:#111; background:#fff; }
.badge img { width:16px; height:16px; }
.card { border:1px solid #eee; border-radius:16px; padding:12px 12px 4px 12px; margin-top:8px; }
.statusbar { border:1px dashed #e5e7eb; border-radius:12px; padding:8px 12px; margin:8px 0 4px 0; display:flex; justify-content:space-between; align-items:center; font-size:0.92rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

st.title("Vigil-(Value_Investment_Graham_Intelligent_Lookup)")

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
def fetch_price_and_name_yf(symbol: str):
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
        # nome leggibile (fallback se NAME non c'è nello sheet)
        name = None
        try:
            info = t.info
            name = info.get("shortName") or info.get("longName")
        except Exception:
            name = None
        return (float(price) if price is not None else None), (name or "")
    except Exception:
        return None, ""

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
    # normalizza timestamp
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

def has_eod_today(ticker: str, today_date: date):
    dfh = load_history()
    if dfh.empty or "Ticker" not in dfh.columns or "Timestamp" not in dfh.columns:
        return False
    dft = dfh[dfh["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if dft.empty: return False
    dft["day"] = pd.to_datetime(dft["Timestamp"], errors="coerce").dt.date
    return any(dft["day"] == today_date)

def make_badge(name: str, url: str, domain: str) -> str:
    """Piccola pill con favicon + nome link"""
    if not url: return ""
    favicon = f"https://www.google.com/s2/favicons?sz=64&domain={domain}"
    return f"<a class='badge' href='{url}' target='_blank' rel='noopener'><img src='{favicon}'/> {name}</a>"

# link autogenerati (nessuna modifica allo sheet necessaria)
def link_yahoo(t):      return f"https://finance.yahoo.com/quote/{t}" if t else ""
def link_investing(t):  return f"https://www.google.com/search?q=Investing+{t}" if t else ""
def link_morning(t):    return f"https://www.google.com/search?q=Morningstar+{t}" if t else ""

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

    # opzionale Nome
    if 0 <= idx_name < ncols:
        df["Name_raw"] = [row[idx_name] if idx_name < len(row) else "" for row in data]
    else:
        df["Name_raw"] = ""

    # togli righe completamente vuote
    mask_nonempty = (df["Ticker_raw"].astype(str).str.strip()!="") | \
                    (df["EPS_raw"].astype(str).str.strip()!="")   | \
                    (df["BVPS_raw"].astype(str).str.strip()!="")
    df = df[mask_nonempty].reset_index(drop=True)

    # normalizza
    df["Ticker"]   = df["Ticker_raw"].astype(str).str.strip().str.upper()
    df["Name"]     = df["Name_raw"].astype(str).str.strip()
    df["EPS"]      = df["EPS_raw"].apply(to_number)
    df["BVPS"]     = df["BVPS_raw"].apply(to_number)
    df["GN_sheet"] = df["GN_sheet_raw"].apply(to_number)

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
# SIDEBAR: toggle modalità
# =========================
with st.sidebar:
    try:
        is_public = st.toggle("🌍 Modalità pubblica", value=False, help="Disattiva funzioni admin e messaggi tecnici.")
    except Exception:
        is_public = st.checkbox("🌍 Modalità pubblica", value=False)
    st.caption("In pubblica: auto-refresh 60s, nessun Debug o messaggi tecnici.")

# =========================
# BUTTON: refresh (unico)
# =========================
col_btn, _ = st.columns([1,3])
with col_btn:
    if st.button("🔄 Aggiorna pagina"):
        st.cache_data.clear()
        st.rerun()

# Auto-refresh discreto (solo Pubblica)
if is_public:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# =========================
# STATUS BAR (ora locale)
# =========================
now_rome = datetime.now(ZoneInfo("Europe/Rome"))
st.markdown(
    f"<div class='statusbar'>"
    f"<div>🕒 Ultimo refresh: {now_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}</div>"
    f"<div>Modalità: {'Pubblica' if is_public else 'Privata'} • Auto-refresh: {'60s' if is_public else 'off'}</div>"
    f"</div>", unsafe_allow_html=True
)

# =========================
# FUNZIONI admin: riscrivere GN su Sheet
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

# =========================
# UI + Ricerca (Ticker o Nome)
# =========================
df, meta = load_fundamentals_by_letter()
if df.empty or df["Ticker"].isna().all():
    st.warning("Nessun dato utile. Controlla che il foglio contenga Ticker(A), EPS(B), BVPS(C), Graham(D).")
else:
    # Barra di ricerca
    q = st.text_input("🔎 Cerca (Ticker o Nome)…", placeholder="es. ENEL.MI o Enel")
    dff = df.copy()
    if q and str(q).strip():
        qn = str(q).strip().lower()
        dff = dff[(dff["Ticker"].str.lower().str.contains(qn)) | (dff["Name"].str.lower().str.contains(qn))]

    # Etichette "TICKER — Nome"
    def _mk_label(row):
        name = row["Name"].strip()
        return f"{row['Ticker']} — {name}" if name else f"{row['Ticker']}"

    options = dff["Ticker"].tolist()
    labels  = dff.apply(_mk_label, axis=1).tolist()
    if not options:
        st.info("Nessun risultato. Prova a cercare un altro Ticker/Nome.")
        st.stop()

    label_to_ticker = {lab: tick for lab, tick in zip(labels, options)}
    selected_label = st.selectbox("Scegli il titolo", options=labels, index=0)
    tick = label_to_ticker[selected_label]

    # Riga selezionata
    row = df[df["Ticker"] == tick].iloc[0].to_dict()
    eps_val    = row.get("EPS")
    bvps_val   = row.get("BVPS")
    gn_sheet   = row.get("GN_sheet")
    nome_sheet = (row.get("Name") or "").strip()

    # Prezzo + (eventuale) nome da Yahoo come fallback
    symbol = normalize_symbol(tick)
    price_live, yf_name = fetch_price_and_name_yf(symbol)
    nice_name = nome_sheet or yf_name or ""  # preferisci il nome da Sheet, poi Yahoo

    # Formula interna (mostrata)
    gn_formula = gn_formula_225(eps_val, bvps_val)

    # Margine % vs GN_sheet
    margin_pct = None
    if price_live is not None and gn_sheet is not None and gn_sheet > 0:
        margin_pct = (1 - (price_live / gn_sheet)) * 100

    # Header con nome + badges link (autogenerati)
    st.markdown(f"### {nice_name}  \n`{tick}`")
    def badge_row(t):
        b = "<div class='badge-row'>"
        b += make_badge("Yahoo Finance", f"https://finance.yahoo.com/quote/{t}", "finance.yahoo.com")
        b += make_badge("Investing",     f"https://www.google.com/search?q=Investing+{t}", "it.investing.com")
        b += make_badge("Morningstar",   f"https://www.google.com/search?q=Morningstar+{t}", "morningstar.com")
        b += "</div>"
        return b
    st.markdown(badge_row(tick), unsafe_allow_html=True)

    # Card metriche
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
        st.caption(f"Fonte: Yahoo – {symbol}")
    with c2:
        st.metric("Numero di Graham (da sheet)", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
        st.caption(f"Colonna: {meta.get('gn_letter')}")
    with c3:
        if margin_pct is not None:
            label = "✅ Sottovalutata" if margin_pct > 0 else "❌ Sopravvalutata"
            st.metric("Margine di sicurezza", f"{margin_pct:.2f}%", label)
        else:
            st.metric("Margine di sicurezza", "n/d")
    st.markdown("</div>", unsafe_allow_html=True)

    # === Snapshot EOD automatico per il TICKER selezionato ===
    # Esegui tra le 17:30 e le 17:50 Europe/Rome, se non già presente per oggi
    close_time = dtime(hour=17, minute=30)
    if is_public and (now_rome.time() >= close_time):
        if not has_eod_today(tick, now_rome.date()):
            try:
                append_history_row(now_rome.strftime("%Y-%m-%d %H:%M:%S"), tick, price_live, eps_val, bvps_val, gn_sheet, "Auto EOD")
                st.success("📌 Snapshot EOD creato automaticamente per il titolo selezionato.")
                st.cache_data.clear()
            except Exception as e:
                # in pubblica evitiamo messaggi tecnici
                pass

    # === Tabs: Dettaglio / Storico ===
    tab1, tab2 = st.tabs(["Dettaglio", "Storico"])
    with tab1:
        # The GN Formula
        st.markdown("### The GN Formula")
        if gn_formula is not None:
            st.code(f"√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_formula:.4f}")
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")

        # area admin (solo Privata)
        if not is_public:
            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                if st.button("✍️ Riscrivi Graham# su Sheet (22,5)"):
                    try:
                        gn_series = compute_gn_series(df)
                        write_gn_to_sheet(ws_fund, gn_series, gn_letter=GN_LETTER)
                        st.success("Colonna Graham# riscritta con i valori corretti (22,5×EPS×BVPS).")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore durante la riscrittura: {e}")
            with colB:
                if st.button("📦 Esegui snapshot EOD per TUTTI i titoli adesso"):
                    try:
                        for _, r in df.iterrows():
                            _tick = r["Ticker"]
                            _eps  = r["EPS"]
                            _bvps = r["BVPS"]
                            _gn   = r["GN_sheet"]
                            if not _tick: 
                                continue
                            price, _ = fetch_price_and_name_yf(normalize_symbol(_tick))
                            append_history_row(now_rome.strftime("%Y-%m-%d %H:%M:%S"), _tick, price, _eps, _bvps, _gn, "Admin EOD All")
                        st.success("Snapshot EOD creato per tutti i titoli.")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Errore snapshot massivo: {e}")

        # Debug (solo Privata)
        if not is_public:
            with st.expander("🔎 Debug colonne & valori"):
                st.write(pd.DataFrame({
                    "Campo": ["Ticker_letter","EPS_letter","BVPS_letter","GN_letter","Name_letter",
                              "Ticker_raw","Name_raw","EPS_raw","BVPS_raw","GN_sheet_raw",
                              "EPS_parsed","BVPS_parsed","GN_sheet_parsed","GN_formula_22_5","Nome_finale"],
                    "Valore": [
                        meta.get("ticker_letter"), meta.get("eps_letter"), meta.get("bvps_letter"), meta.get("gn_letter"), meta.get("name_letter"),
                        row.get("Ticker_raw"), row.get("Name_raw"), row.get("EPS_raw"), row.get("BVPS_raw"), row.get("GN_sheet_raw"),
                        eps_val, bvps_val, gn_sheet, gn_formula, nice_name
                    ]
                }))

    with tab2:
        # STORICO: tabella + filtri + download + grafico
        dfh = load_history()
        if dfh.empty or "Ticker" not in dfh.columns:
            st.info("Nessuno storico disponibile.")
        else:
            dft = dfh[dfh["Ticker"].astype(str).str.upper() == tick.upper()].copy()
            if dft.empty:
                st.info("Nessuno storico disponibile per questo titolo.")
            else:
                # filtri data
                min_day = pd.to_datetime(dft["Timestamp"]).dt.date.min()
                max_day = pd.to_datetime(dft["Timestamp"]).dt.date.max()
                start, end = st.date_input("Intervallo date", value=(min_day, max_day), min_value=min_day, max_value=max_day)
                if isinstance(start, date) and isinstance(end, date) and start <= end:
                    dft = dft[(pd.to_datetime(dft["Timestamp"]).dt.date >= start) &
                              (pd.to_datetime(dft["Timestamp"]).dt.date <= end)]
                dft = dft.sort_values("Timestamp")

                # mostra tabella
                show_cols = [c for c in ["Timestamp","Ticker","Price","EPS","BVPS","Graham","Fonte"] if c in dft.columns]
                st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

                # download CSV
                csv = dft[show_cols].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Scarica CSV", data=csv, file_name=f"storico_{tick}.csv", mime="text/csv")

                # grafico semplice (Prezzo e GN se presenti)
                if "Price" in dft.columns and "Graham" in dft.columns:
                    plot_df = dft[["Timestamp","Price","Graham"]].dropna()
                    plot_df = plot_df.set_index("Timestamp")
                    st.line_chart(plot_df, use_container_width=True)
