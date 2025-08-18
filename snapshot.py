# snapshot.py — salva snapshot di TUTTI i titoli su tab "Storico"
import os, json, math
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread, yfinance as yf
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

SHEET_ID = os.environ["SHEET_ID"]
FUND_TAB = os.environ.get("FUND_TAB", "Fondamentali")
HIST_TAB = os.environ.get("HIST_TAB", "Storico")

def auth():
    sa_json = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(sa_json, scopes=SCOPES)
    return gspread.authorize(creds)

def price_live_or_close(ticker):
    t = yf.Ticker(ticker)
    # prova fast_info
    try:
        p = float(t.fast_info.last_price)
        if p and p > 0 and math.isfinite(p):
            return p
    except Exception:
        pass
    # fallback intraday
    try:
        h = t.history(period="1d", interval="1m")
        if len(h) > 0:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    # fallback close
    try:
        h = t.history(period="5d", interval="1d")
        if len(h) > 0:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    tz = ZoneInfo("Europe/Rome")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    gc = auth()
    sh = gc.open_by_key(SHEET_ID)
    fund = sh.worksheet(FUND_TAB)
    hist = sh.worksheet(HIST_TAB)

    rows = fund.get_all_records()
    out = []
    for r in rows:
        t = str(r.get("Ticker","")).strip().upper()
        if not t:
            continue
        eps  = to_float(r.get("EPS"))
        bvps = to_float(r.get("BVPS"))
        gn   = to_float(r.get("Graham"))
        px   = price_live_or_close(t)
        delta  = (px - gn) if (px is not None and gn is not None) else ""
        margin = (1 - (px/gn))*100 if (px is not None and gn not in (None,0)) else ""
        out.append([now, t, px or "", eps or "", bvps or "", gn or "", delta, margin, "Cron"])
    if out:
        hist.append_rows(out, value_input_option="USER_ENTERED")
        print(f"Appended {len(out)} rows to {HIST_TAB}")

if __name__ == "__main__":
    main()
