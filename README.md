# Vigil – Value Investment Graham Intelligent Lookup

Web-app Streamlit collegata a Google Sheets che mostra:
- Prezzo live (Yahoo Finance)
- Numero di Graham (22.5 × EPS × BVPS)
- Margine di sicurezza
- Snapshot su tab `Storico`

## Avvio rapido

1. Crea una Service Account Google e condividi lo Sheet con il suo indirizzo (Editor).
2. Deploy su Streamlit Community Cloud e incolla i **Secrets** (vedi sotto).
3. Per snapshot giornaliero, usa GitHub Actions e imposta i **Repository Secrets**.

### Secrets (Streamlit → Settings → Secrets)

```toml
[gcp_service_account]
type = "service_account"
project_id = "<PROJECT_ID>"
private_key_id = "<PRIVATE_KEY_ID>"
private_key = "-----BEGIN PRIVATE KEY-----\n<KEY_WITH_ESCAPED_NEWLINES>\n-----END PRIVATE KEY-----\n"
client_email = "<SERVICE_ACCOUNT_EMAIL>"
client_id = "<CLIENT_ID>"
token_uri = "https://oauth2.googleapis.com/token"

[gsheet]
sheet_id = "<SHEET_ID>"
fundamentals_tab = "Fondamentali"
history_tab = "Storico"

[app]
public_mode = true
admin_access_code = "Darazz1920"
default_suffix = ".MI"
```

### Repository Secrets (GitHub → Settings → Secrets and variables → Actions)
- `GCP_SERVICE_ACCOUNT_JSON` → incolla l'intero JSON della service account.
- `SHEET_ID` → il tuo Sheet ID.
