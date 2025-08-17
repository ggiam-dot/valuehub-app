import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup", layout="centered")

# -------------------- Helpers: Secrets & Validazioni --------------------
def get_secret(section: str, key: str, default=None):
    sect = st.secrets.get(section, {})
    if not isinstance(sect, dict):
        return default
    return sect.get(key, default)

def validate_secrets():
    problems = []

    # Sezioni richieste
    gcp = st.secrets.get("gcp_service_account")
    gsh = st.secrets.get("gsheet")
    app_cfg = st.secrets.get("app", {})

    if not isinstance(gcp, dict):
        problems.append("Manca la sezione [gcp_service_account] nei Secrets.")
    else:
        # Campi minimi richiesti per le credenziali
        for k in ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]:
            if not gcp.get(k):
                problems.append(f"[gcp_service_account].{k} mancante.")

    if not isinstance(gsh, dict):
        problems.append("Manca la sezione [gsheet] nei Secrets.")
    else:
        for k in ["sheet_id","fundamentals_tab","history_tab"]:
            if not gsh.get(k):
                problems.append(f"[gsheet].{k} mancante.")

    # app (facoltativa ma consigliata)
    if not isinstance(app_cfg, dict):
        problems.append("Sezione [app] mancante (verrà usata config di default).")

    return problems

def secrets_status_panel():
    st.subheader("Verifica configurazione")
    problems = validate_secrets()
    if problems:
        for p in problems:
            st.error("❌ " + p)
        st.info("Vai su Streamlit Cloud → *Manage app* → *Advanced settings* → **Secrets** e incolla il template che ti ho dato (con i tuoi valori).")
        st.stop()
    else:
        st.success("✅ Secrets OK")

# -------------------- Config from Secrets (con fallback sicuri) --------------------
APP_CFG = st.secrets.get("app", {})
APP_PUBLIC = APP_CFG.get("public_mode", True)
ADMIN_CODE = APP_CFG.get("admin_access_code", "")
DEFAULT_SUFFIX = APP_CFG.get("default_suffix", ".MI")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_client_and_ws():
    gcp = st.secrets.get("gcp_service_account", {})
    try:
        creds = Credentials.from_service_account_info(gcp, scopes=SCOPES)
    except Exception as e:
        st.error("❌ Credenziali Google non valide. Controlla `private_key` (deve avere le \\n) e i campi del JSON.")
        st.exception(e)
        st.stop()

    try:
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["gsheet"]["sheet_id"])
        fond = sh.worksheet(st.secrets["gsheet"]["fundamentals_tab"])
        hist = sh.worksheet(st.secrets["gsheet"]["history_tab"])
        return gc, sh, fond, hist
    except gspread.exceptions.APIError as e:
        st.error("❌ Permessi Google Sheets: assicurati di aver **condiviso lo Sheet in Editor** con la service account.")
        st.code(st.secrets["gcp_service_account"].get("client_email","<missing>"))
        st.exception(e)
        st.stop()
    except gspread.exceptions.WorksheetNotFound as e:
        st.error("❌ Nome tab errato. Devono esistere esattamente: 'Fondamentali' e 'Storico'.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("❌ Errore di connessione a Google Sheets.")
        st.exception(e)
        st.stop()

@st.cache_data(ttl=300)
def load_fundamentals():
    _, _, fond, _ = get_client_and_ws()
    rows = fond.get_all_records()
    df = pd.DataFrame(rows)
    # Colonne attese
    for c in ["Ticker", "EPS", "BVPS"]:
        if c not in df.columns: df[c] = np.nan
    # Se manca 'Graham' in Fondamentali lo calcoliamo (fallback)
    if "Graham" not in df.columns:
        df["Graham"] = np.sqrt(
            22.5 *
            df["EPS"].astype(float).clip(lower=0) *
            df["BVPS"].astype(float).clip(lower=0)
        )
    for c in ["Name", "InvestingURL", "MorningstarURL", "ISIN"]:
        if c not in df.columns: df[c] = ""
    return df

def latest_price(ticker: str):
    t = yf.Ticker(ticker)
    # fast path
    try:
        p = float(t.fast_info.last_price)
        if p and p > 0:
            return p
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
    code = st.text_input("Admin code per azioni di scrittura", type="password", key="write_code")
    return (code == ADMIN_CODE) and (ADMIN_CODE != "")

def append_history(hist_ws, row_list):
    hist_ws.append_row(row_list, value_input_option="USER_ENTERED")

# -------------------- UI --------------------
st.title("Vigil – Value Investment Graham Intelligent Lookup")
st.caption("Prezzo live (Yahoo), Numero di Graham, Margine di Sicurezza e snapshot su Google Sheets.")

# 1) Verifica Secrets e accesso a Google prima di procedere
secrets_status_panel()

# 2) Carica Fondamentali
df = load_fundamentals()
tickers = sorted([t for t in df["Ticker"].dropna().astype(str).unique() if t.strip()])

col1, col2 = st.columns([3, 2])
with col1:
    sel = st.selectbox("Seleziona Ticker", options=[""] + tickers, index=0, help="Dai dati del foglio 'Fondamentali'")
with col2:
    manual = st.text_input("…oppure digita un ticker", placeholder="es. ENEL.MI")

ticker = manual.strip() or sel.strip()

if ticker:
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

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**ISIN**: {row['ISIN'].iloc[0] if not row.empty else '—'}")
    with c2:
        inv = row["InvestingURL"].iloc[0] if not row.empty else ""
        if inv: st.markdown(f"[Scheda Investing]({inv})")
    with c3:
        ms = row["MorningstarURL"].iloc[0] if not row.empty else ""
        if ms: st.markdown(f"[Scheda Morningstar]({ms})")

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

with st.expander("Vedi tabella Fondamentali (read-only)"):
    st.dataframe(df, use_container_width=True)
