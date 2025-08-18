import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz, math
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Vigil – Value Investment Graham Lookup", layout="wide")

# =============== CONFIG / SECRETS CHECK =================
def _problems():
    probs = []
    gcp = st.secrets.get("gcp_service_account")
    gsh = st.secrets.get("gsheet")
    if not isinstance(gcp, dict): probs.append("Manca la sezione [gcp_service_account] nei Secrets.")
    else:
        for k in ["type","project_id","private_key_id","private_key","client_email","client_id","token_uri"]:
            if not gcp.get(k): probs.append(f"[gcp_service_account].{k} mancante.")
    if not isinstance(gsh, dict): probs.append("Manca la sezione [gsheet] nei Secrets.")
    return probs

def _require_secrets():
    probs = _problems()
    if probs:
        st.title("Vigil – Value Investment Graham Intelligent Lookup")
        st.subheader("Verifica configurazione")
        for p in probs: st.error("❌ " + p)
        st.info("Vai su **Manage app → Advanced settings → Secrets** e incolla il template.")
        st.stop()

_require_secrets()

APP_PUBLIC = st.secrets.get("app", {}).get("public_mode", True)
ADMIN_CODE = st.secrets.get("app", {}).get("admin_access_code", "")
DEFAULT_SUFFIX = st.secrets.get("app", {}).get("default_suffix", ".MI")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

def get_client_and_ws():
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["gsheet"]["sheet_id"])
    fond = sh.worksheet(st.secrets["gsheet"]["fundamentals_tab"])
    hist = sh.worksheet(st.secrets["gsheet"]["history_tab"])
    return gc, sh, fond, hist

@st.cache_data(ttl=300)
def load_fundamentals():
    _, _, fond, _ = get_client_and_ws()
    df = pd.DataFrame(fond.get_all_records())
    # colonne attese
    for c in ["Ticker","EPS","BVPS"]: 
        if c not in df.columns: df[c] = np.nan
    if "Graham" not in df.columns:
        # fallback: calcolo da EPS/BVPS
        df["Graham"] = np.sqrt(
            22.5 * df["EPS"].astype(float).clip(lower=0) * df["BVPS"].astype(float).clip(lower=0)
        )
    for c in ["Name","InvestingURL","MorningstarURL","ISIN"]:
        if c not in df.columns: df[c] = ""
    return df

# =============== PRICES / FORMULAE ======================
def latest_price(ticker: str):
    t = yf.Ticker(ticker)
    # 1) fast_info
    try:
        p = float(t.fast_info.last_price)
        if p and p > 0 and math.isfinite(p): return p
    except Exception:
        pass
    # 2) 1m intraday
    try:
        h = t.history(period="1d", interval="1m")
        if len(h) > 0:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    # 3) daily close
    try:
        h = t.history(period="1d")
        if len(h) > 0:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

def compute_gn(eps, bvps):
    try:
        eps = float(eps); bvps = float(bvps)
        if eps > 0 and bvps > 0: return float(math.sqrt(22.5 * eps * bvps))
    except Exception:
        pass
    return None

def compute_mos(gn, price):
    if gn and gn > 0 and price is not None:
        return (gn - price) / gn
    return None

# =============== UI =====================================
st.title("📈 Vigil – Value Investment Graham Intelligent Lookup")
tabs = st.tabs(["🇮🇹 Analisi", "📜 Storico"])

df = load_fundamentals()
tickers = sorted([t for t in df["Ticker"].dropna().astype(str).unique() if t.strip()])

with tabs[0]:
    col_sel, col_manual = st.columns([3,2])
    with col_sel:
        sel = st.selectbox("Scegli il Ticker", options=[""]+tickers, index=0)
    with col_manual:
        manual = st.text_input("…oppure digita un ticker", placeholder="ENEL.MI / AAPL")
    ticker = (manual.strip() or sel.strip())

    # ---- refresh controls
    r1, r2, r3 = st.columns([1,1,3])
    with r1:
        if st.button("🔄 Aggiorna ora"):
            st.cache_data.clear()
            st.experimental_rerun()
    with r2:
        auto = st.toggle("Auto-refresh (60s)", value=False)
        if auto:
            try:
                st.autorefresh(interval=60_000, key="auto")
            except Exception:
                pass

    # ---- admin mode
    am1, am2 = st.columns([2,3])
    with am1:
        admin_toggle = st.toggle("🛠 Modalità amministratore", value=False, help="Richiede admin code per scrivere sullo Sheet")
    with am2:
        can_write = False
        if admin_toggle:
            code = st.text_input("Admin code", type="password", key="admin_code")
            can_write = (code == ADMIN_CODE and ADMIN_CODE != "")

    if ticker:
        row = df[df["Ticker"].astype(str) == ticker].head(1)
        eps = float(row["EPS"].iloc[0]) if not row.empty and pd.notnull(row["EPS"].iloc[0]) else None
        bvps = float(row["BVPS"].iloc[0]) if not row.empty and pd.notnull(row["BVPS"].iloc[0]) else None
        gn_sheet = float(row["Graham"].iloc[0]) if not row.empty and pd.notnull(row["Graham"].iloc[0]) else None

        price = latest_price(ticker)
        gn_calc = compute_gn(eps, bvps)
        gn = gn_sheet if gn_sheet else gn_calc
        mos = compute_mos(gn, price)

        st.markdown(f"### **{ticker}** — {row['Name'].iloc[0] if not row.empty and row['Name'].iloc[0] else ''}")
        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Prezzo live", f"{price:.2f}" if price is not None else "—")
        meta2.metric("Graham#", f"{gn:.2f}" if gn else "—")
        if mos is not None:
            label = "Sottovalutata" if mos >= 0 else "Sopravvalutata"
            meta3.metric("Margine di sicurezza", f"{mos*100:.2f}%")
            st.markdown(f"**{label}**")
        else:
            meta3.metric("Margine di sicurezza", "—")

        with st.expander("The GN Formula (Applied)"):
            st.markdown(f"√(22.5 × {eps if eps is not None else 'EPS'} × {bvps if bvps is not None else 'BVPS'}) = **{f'{gn:.4f}' if gn else '—'}**")
            st.caption("EPS e BVPS dal foglio; coefficiente 22.5 (Graham)")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            inv = row["InvestingURL"].iloc[0] if not row.empty else ""
            if inv: st.markdown(f"[Investing]({inv})")
        with c2:
            ms = row["MorningstarURL"].iloc[0] if not row.empty else ""
            if ms: st.markdown(f"[Morningstar]({ms})")
        with c3:
            st.write(f"ISIN: {row['ISIN'].iloc[0] if not row.empty else '—'}")

        st.divider()
        st.subheader("📌 Snapshot su Storico")
        if can_write:
            if st.button("💾 Salva snapshot TICKER"):
                _, _, _, hist_ws = get_client_and_ws()
                tz = pytz.timezone("Europe/Rome")
                now = dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
                eps_v = eps if eps is not None else ""
                bvps_v = bvps if bvps is not None else ""
                gn_v = gn if gn is not None else ""
                delta = (gn - price) if (gn and price is not None) else ""
                mos_v = ((gn - price) / gn) if (gn and price is not None and gn>0) else ""
                row_out = [now, ticker, price or "", eps_v, bvps_v, gn_v, "Yahoo Finance", price or "", gn_v, delta, mos_v]
                hist_ws.append_row(row_out, value_input_option="USER_ENTERED")
                st.success("Snapshot salvato ✅")

            if st.button("🗂️ Salva snapshot di TUTTI i titoli"):
                _, _, _, hist_ws = get_client_and_ws()
                tz = pytz.timezone("Europe/Rome")
                now = dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
                rows_out = []
                for _, r in df.iterrows():
                    tck = str(r.get("Ticker","")).strip()
                    if not tck: continue
                    px = latest_price(tck)
                    eps_i = r.get("EPS",""); bvps_i = r.get("BVPS","")
                    gn_sheet_i = r.get("Graham","")
                    gn_calc_i = compute_gn(eps_i, bvps_i)
                    gn_i = float(gn_sheet_i) if str(gn_sheet_i).strip() not in ["","None"] else (gn_calc_i if gn_calc_i else None)
                    if px is None and gn_i is None: continue
                    delta = (gn_i - px) if (gn_i is not None and px is not None) else ""
                    mos_i = ((gn_i - px)/gn_i) if (gn_i is not None and px is not None and gn_i>0) else ""
                    rows_out.append([now, tck, px or "", eps_i or "", bvps_i or "", gn_i or "", "Yahoo Finance", px or "", gn_i or "", delta, mos_i])
                if rows_out:
                    hist_ws.append_rows(rows_out, value_input_option="USER_ENTERED")
                    st.success(f"Snapshot di {len(rows_out)} titoli salvato ✅")
                else:
                    st.warning("Nessun ticker valido da salvare.")
        else:
            st.info("Attiva 'Modalità amministratore' e inserisci l'admin code per scrivere sullo Sheet.")

with tabs[1]:
    st.caption("Ultimi snapshot da tab **Storico**")
    try:
        _, _, _, hist_ws = get_client_and_ws()
        hist = pd.DataFrame(hist_ws.get_all_records())
        if not hist.empty:
            colf = st.columns([1,1,2,2,2,2,2,2,2,2,2])
            tick_filter = st.selectbox("Ticker", ["(tutti)"] + sorted(hist["Ticker"].dropna().unique().tolist()))
            if tick_filter != "(tutti)":
                hist = hist[hist["Ticker"] == tick_filter]
            hist = hist.tail(500).iloc[::-1].reset_index(drop=True)
            st.dataframe(hist, use_container_width=True)
        else:
            st.info("La tab Storico è vuota.")
    except Exception as e:
        st.warning("Impossibile leggere la tab Storico (controlla permessi/nomi).")
