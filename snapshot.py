# snapshot.py – salva snapshot su "Storico" (chiusura), calcola GN da EPS/BVPS se possibile
import os, json, math, re
from datetime import datetime
from zoneinfo import ZoneInfo
import gspread, yfinance as yf
from google.oauth2.service_account import Credentials
import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

SHEET_ID = os.environ["SHEET_ID"]
FUND_TAB = os.environ.get("FUND_TAB", "Fondamentali")
HIST_TAB = os.environ.get("HIST_TAB", "Storico")
YF_SUFFIX = os.environ.get("YF_SUFFIX", ".MI")

COL_T, COL_E, COL_B, COL_G = 0,1,2,3  # A,B,C,D

def auth():
    sa = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

def norm_sym(s):
    s = (s or "").strip().upper()
    return s if "." in s else s + YF_SUFFIX

def to_num(x):
    try:
        if x is None or str(x).strip()=="":
            return None
        s = str(x).replace("€","").replace("%","").replace("\u00A0","").strip()
        s = re.sub(r"[^0-9\-,\.]","",s)
        if s in ("","-",".",","): return None
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".","").replace(",",".")
            else:
                s = s.replace(",","")
        elif "," in s:
            s = s.replace(",",".")
        return float(s)
    except:
        return None

def close_price(sym):
    t = yf.Ticker(sym)
    # chiusura più recente
    try:
        h = t.history(period="10d", interval="1d")["Close"].dropna()
        if len(h) > 0:
            return float(h.iloc[-1])
    except: pass
    return None

def gn_from_eps_bvps(eps, bvps):
    if eps and bvps and eps>0 and bvps>0:
        return float((22.5*eps*bvps) ** 0.5)
    return None

def main():
    tz = ZoneInfo("Europe/Rome")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    gc = auth()
    sh = gc.open_by_key(SHEET_ID)
    fund = sh.worksheet(FUND_TAB)
    hist = sh.worksheet(HIST_TAB)

    rows = fund.get_all_values()[1:]  # skip header
    out = []
    for r in rows:
        t = (r[COL_T] if len(r)>COL_T else "").strip().upper()
        if not t: continue
        eps  = to_num(r[COL_E] if len(r)>COL_E else None)
        bvps = to_num(r[COL_B] if len(r)>COL_B else None)
        gn_sheet = to_num(r[COL_G] if len(r)>COL_G else None)

        gn = gn_from_eps_bvps(eps,bvps) or gn_sheet
        px = close_price(norm_sym(t))

        delta  = (px - gn) if (px is not None and gn is not None) else None
        margin = (1 - (px/gn))*100 if (px is not None and gn not in (None,0)) else None

        out.append([
            now, t,
            ("" if px is None else float(px)),
            ("" if eps is None else float(eps)),
            ("" if bvps is None else float(bvps)),
            ("" if gn is None else float(gn)),
            ("" if delta is None else float(delta)),
            ("" if margin is None else float(margin)),
            "Cron (close)"
        ])

    if out:
        hist.append_rows(out, value_input_option="USER_ENTERED")
        print(f"OK snapshot {len(out)} righe")

if __name__ == "__main__":
    main()
