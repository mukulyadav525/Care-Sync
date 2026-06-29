# M-Health

A privacy-first health platform for collecting and analysing physiological
signals (PPG, GSR, actigraphy). The project is split into three clearly
separated layers plus a small device-side ingestion service.

```
.
├── frontend/           # Next.js 16 + React 19 web app (UI)
├── backend/            # Django 6 + DRF REST API (business logic, auth, HL7)
├── database/           # SQLite database file (db.sqlite3)
└── device-ingestion/   # Standalone Flask scripts that receive raw sensor data
```

## Layers

### `frontend/` — Web UI
Next.js app (App Router) that talks to the backend over `/api`. Pages:
`login`, `dashboard`, `files`, the PPG `visualization` view, and the
device **Signal Portal** (`/portal`). Styling uses the **CareVibe** emerald
theme defined in `src/app/globals.css`.

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

The API base URL is configured in `src/lib/api.ts` (`http://127.0.0.1:8000/api`).

### `backend/` — REST API
Django REST Framework, JWT auth (SimpleJWT), API-only (no server-rendered
templates). Endpoints are grouped under `/api/` in `mHealth/urls.py` (Care-Sync)
(`auth`, `files`, `visualization`, `hl7`). App code lives in `login/`.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver   # http://127.0.0.1:8000
```

### Signal Portal (`/portal`)
Visualises wearable **Empatica-E4** session data. Each *session* is a folder
inside `backend/Users/<username>/` containing the device export files:

| File | Signal | Portal view |
|------|--------|-------------|
| `HR.csv` | Heart rate (bpm) | line chart + min/avg/max |
| `EDA.csv` | Skin conductance (µS) | line chart + min/avg/max |
| `TEMP.csv` | Skin temperature (°C) | line chart + min/avg/max |
| `ACC.csv` | Accelerometer x/y/z | movement magnitude (g) |
| `BVP.csv` | Blood volume pulse | raw waveform sample |
| `IBI.csv` | Inter-beat intervals | HRV (RMSSD / SDNN / mean HR) |
| `tags*.csv` | Event markers | event count |
| `info.txt` | Metadata | shown verbatim |

The portal aggregates continuous signals **per minute / hour / day** (selectable),
shows headline metrics, per-signal charts and a day-by-day averages table.

Backend endpoints (`login/views/device.py`):
- `GET /api/device/sessions/` — list sessions for the user (all users if admin)
- `GET /api/device/sessions/<owner>/<name>/?granularity=hour` — full parsed data

Generate a demo session for local testing:

```bash
cd backend
.venv/bin/python scripts/generate_sample_e4.py <username> <session_name> <hours>
# e.g. python scripts/generate_sample_e4.py mukul session_demo 3
```

### `database/`
Holds the SQLite database (`db.sqlite3`). The backend resolves this path
automatically (`BASE_DIR.parent / 'database' / 'db.sqlite3'`).

### `device-ingestion/`
Standalone Flask servers used by the wearable/hardware to stream raw sensor
CSV data. These are independent of the Django web stack and are run directly
on the data-collection machine.

- `start_session_server.py` — fixed upload directory
- `user_session_server.py` — prompts for a username and writes per-user files

```bash
python device-ingestion/start_session_server.py   # listens on :5000
```

## Connected devices
Register wearables under **Devices** (`/devices`). Each registration returns a
one-time **device key**. The device / ingestion side posts heartbeats so the
portal can show live online/offline status, battery and firmware:

```
POST /api/devices/heartbeat/    header: X-Device-Key: <key>
body: { "battery": 88, "firmware": "1.0.0", "session": "session_demo" }
```

A device is shown **online** if it sent a heartbeat in the last 120 s. See
[device-ingestion/heartbeat.py](device-ingestion/heartbeat.py) for a tiny
example the ingestion script can call.

## Security model
This is medical data, so the backend is locked down:

- **No anonymous data access.** Every data / file / visualization / device
  endpoint requires JWT auth; the only public endpoints are health, contact,
  and the auth/OTP endpoints themselves.
- **Two-factor authentication.** Both sign-up and sign-in require an emailed
  6-digit code (`EmailOTP`): hashed at rest, expires in 10 min, attempt-limited.
- **No plaintext passwords.** Sign-up holds only a hashed password until the
  code is verified; passwords are checked against Django's strength validators.
- **Rate limiting.** Login / OTP / heartbeat endpoints are IP-throttled.
- **Token revocation.** Logout blacklists the refresh token
  (`rest_framework_simplejwt.token_blacklist`).
- **Path-traversal safe.** All user-file access goes through `safe_join()`
  (`login/security.py`), and ownership (IDOR) checks scope every record to its
  owner (admins excepted).
- **Production hardening.** With `DEBUG=False`: secret key required from env,
  `ALLOWED_HOSTS` enforced, secure cookies, HSTS, SSL redirect.

In development (`DEBUG=True`) with no SMTP configured, OTP emails are printed to
the console and the code is echoed in the API response (`dev_otp`) so the flow
stays testable.

## Configuration
Backend secrets (Django secret key, email, Google OAuth) are read from
environment variables — see [backend/.env.example](backend/.env.example).
There are **no committed secret fallbacks**; outside `DEBUG`, a missing
`DJANGO_SECRET_KEY` stops startup. Never commit `.env` or `credentials.json`
(both are gitignored).
