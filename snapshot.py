# snapshot.py — salva snapshot di TUTTI i titoli su tab "Storico" senza usare header
import os, json, math, re
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread, yfinance as yf
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

SHEET_ID = os.environ["SHEET_ID"]
FUND_TAB = os.environ.get("FUND_TAB", "Fondamentali")
HIST_TAB = os.environ.get("HIST_TAB", "Storico")
YF_SUFFIX = os.environ.get("YF_SUFFIX", ".MI")  # opzionale

# colonne fisse: A=Ticker, B=EPS, C=BVPS, D=Graham
COL_T, COL_E, COL_B, COL_G = 0, 1, 2, 3

def auth():
    sa_json = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(sa_json, scopes=SCOPES)
    return gspread.authorize(creds)

def norm_symbol(s: str):
    s = (s or "").strip().upper()
    return s if "." in s else s + YF_SUFFIX

def to_float(x):
    try:
        if x is None: return None
        s = str(x).strip().replace("\u00A0","")
        s = re.sub(r"[^0-9\-,\.]","",s)
        if s in {"","-","." ,","}: return None
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".","").replace(",",".")
            else:
                s = s.replace(",","")
        elif "," in s:
            s = s.replace(",",".")
        return float(s)
    except: return None

def price_live_or_close(ticker):
    t = yf.Ticker(ticker)
    # fast_info / intraday
    try:
        p = t.fast_info.get("last_price")
        if p and p > 0 and math.isfinite(float(p)): return float(p)
    except: pass
    try:
        h = t.history(period="1d", interval="1m")
        if len(h) > 0: return float(h["Close"].dropna().iloc[-1])
    except: pass
    # close
    try:
        h = t.history(period="5d", interval="1d")
        if len(h) > 0: return float(h["Close"].dropna().iloc[-1])
    except: pass
    return None

def main():
    tz = ZoneInfo("Europe/Rome")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    gc = auth()
    sh = gc.open_by_key(SHEET_ID)
    fund = sh.worksheet(FUND_TAB)
    hist = sh.worksheet(HIST_TAB)

    values = fund.get_all_values()
    if not values or len(values) < 2:
        print("Fondamentali vuoto.")
        return

    data = values[1:]  # salta header (qualsiasi esso sia)
    out = []
    for row in data:
        # proteggi da righe corte
        t_raw = row[COL_T] if len(row) > COL_T else ""
        eps_raw = row[COL_E] if len(row) > COL_E else ""
        bvps_raw= row[COL_B] if len(row) > COL_B else ""
        gn_raw  = row[COL_G] if len(row) > COL_G else ""

        t = (t_raw or "").strip().upper()
        if not t: continue

        eps = to_float(eps_raw)
        bvps= to_float(bvps_raw)
        gn  = to_float(gn_raw)
        px  = price_live_or_close(norm_symbol(t))

        delta  = (px - gn) if (px is not None and gn is not None) else ""
        margin = (1 - (px/gn))*100 if (px is not None and gn not in (None,0)) else ""
        out.append([now, t, px or "", eps or "", bvps or "", gn or "", delta, margin, "Cron"])

    if out:
        hist.append_rows(out, value_input_option="USER_ENTERED")
        print(f"Appended {len(out)} rows to {HIST_TAB}")

if __name__ == "__main__":
    main()
