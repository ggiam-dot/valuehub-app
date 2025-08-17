# snapshot.py
import os, json, datetime as dt, pytz, gspread, yfinance as yf
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

def get_client():
    sa_json = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(sa_json, scopes=SCOPES)
    return gspread.authorize(creds)

def latest_price(ticker):
    t = yf.Ticker(ticker)
    try:
        p = float(t.fast_info.last_price)
        if p and p > 0: return p
    except Exception:
        pass
    try:
        h = t.history(period="1d")
        if len(h) > 0:
            return float(h["Close"][-1])
    except Exception:
        pass
    return None

def compute_gn(eps, bvps):
    try:
        eps = float(eps); bvps = float(bvps)
        if eps > 0 and bvps > 0:
            import math
            return math.sqrt(22.5 * eps * bvps)
    except Exception:
        pass
    return None

def main():
    tz = pytz.timezone("Europe/Rome")
    now = dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    gc = get_client()
    sh = gc.open_by_key(os.environ["SHEET_ID"])
    fund_tab = os.environ.get("FUND_TAB", "Fondamentali")
    hist_tab = os.environ.get("HIST_TAB", "Storico")
    fond = sh.worksheet(fund_tab)
    hist = sh.worksheet(hist_tab)

    rows = fond.get_all_records()
    out = []
    for r in rows:
        ticker = str(r.get("Ticker", "")).strip()
        if not ticker:
            continue
        eps = r.get("EPS", "")
        bvps = r.get("BVPS", "")
        gn_sheet = r.get("Graham", "")
        price = latest_price(ticker)
        gn_calc = compute_gn(eps, bvps)
        gn = float(gn_sheet) if (str(gn_sheet).strip() not in ["", "None"]) else (gn_calc if gn_calc else None)

        if price is None and gn is None:
            continue

        delta = (gn - price) if (gn is not None and price is not None) else ""
        mos = ((gn - price) / gn) if (gn is not None and price is not None and gn > 0) else ""

        row_out = [now, ticker, price or "", eps or "", bvps or "", gn or "", "Yahoo Finance", price or "", gn or "", delta, mos]
        out.append(row_out)

    if out:
        hist.append_rows(out, value_input_option="USER_ENTERED")

if __name__ == "__main__":
    main()
