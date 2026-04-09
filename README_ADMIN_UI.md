# FastService Admin UI

Separate admin application for FastService.

## Run

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api.admin_ui:app --host 0.0.0.0 --port 8020
```

Open:

`http://127.0.0.1:8020`

Or with Docker Compose:

```bash
docker compose up -d --build fastservice-admin
```

## Features

- Service control: start/stop/restart managed `app.api.main`, change workers.
- Model reload: calls `POST /admin/models/reload` on FastService.
- Logs tab: tail of `app.log` and managed service stdout log.
- Debug tab:
  - toggle debug visualization,
  - latest image with bbox + recognized text,
  - table of recognized numbers history.
- Testing tab:
  - choose dataset folder directly in UI (folder picker, no manual path),
  - send selected `images/labels/text` files to backend for test run,
  - supports `images/`, optional `labels/`, optional `text/`,
  - if `labels`/`text` are missing, related metrics are not calculated,
  - view/download reports for previous runs,
  - manual report cleanup (`policy` or `all`).

## Notes

- Managed service stdout is written to `dashboard_data/service_stdout.log`.
- Test run history is stored in `dashboard_data/admin_ui.db`.
- HTML reports are stored in `dashboard_data/test_reports/run_<id>.html`.
- Recognition history is stored by main service in `dashboard_data/recognitions.db`.
- In managed mode on Windows, admin starts FastService with `workers=1` for stability even if config value is higher.

## Testing report cleanup policy

Cleanup defaults are loaded from `app/config/admin_ui_config.json`:

```json
{
  "testing": {
    "reports_cleanup": {
      "enabled": true,
      "max_age_days": 30,
      "keep_last_runs": 100
    }
  }
}
```

Supported environment overrides:

- `ADMIN_UI_REPORTS_CLEANUP_ENABLED`
- `ADMIN_UI_REPORTS_CLEANUP_MAX_AGE_DAYS`
- `ADMIN_UI_REPORTS_CLEANUP_KEEP_LAST_RUNS`

## Global tab visibility

Tabs are loaded from config file:

- `app/config/admin_ui_config.json`

Example:

```json
{
  "ui": {
    "visible_tabs": ["service"]
  }
}
```

Optional custom config path:

- `ADMIN_UI_CONFIG_PATH=/path/to/admin_ui_config.json`

You can still override by environment variable at startup:

- `ADMIN_UI_VISIBLE_TABS=service,about` (has higher priority)

Allowed names: `service`, `logs`, `debug`, `testing`, `about`.
If variable is empty, all tabs are visible.
