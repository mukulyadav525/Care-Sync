"""
Minimal device heartbeat helper for the ingestion side.

Call send_heartbeat() periodically (e.g. once a minute, and on session
start/stop) so the web portal can show the device as online with live
battery / firmware / current-session info.

The device key is created in the web UI under "Devices -> Register device"
and shown only once. Configure it via the DEVICE_KEY environment variable.

    export MHEALTH_API=http://127.0.0.1:8000
    export DEVICE_KEY=<the key from the portal>
    python device-ingestion/heartbeat.py            # one ping
"""
import os
import json
import urllib.request

API = os.environ.get('MHEALTH_API', 'http://127.0.0.1:8000')
DEVICE_KEY = os.environ.get('DEVICE_KEY', '')


def send_heartbeat(battery=None, firmware=None, session=None) -> bool:
    """POST a heartbeat. Returns True on success."""
    if not DEVICE_KEY:
        print('DEVICE_KEY not set — register a device in the portal first.')
        return False
    payload = {k: v for k, v in
               {'battery': battery, 'firmware': firmware, 'session': session}.items()
               if v is not None}
    req = urllib.request.Request(
        f'{API}/api/devices/heartbeat/',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'X-Device-Key': DEVICE_KEY},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print('heartbeat ok:', r.read().decode())
            return True
    except Exception as e:
        print('heartbeat failed:', e)
        return False


if __name__ == '__main__':
    send_heartbeat(battery=100, firmware='1.0.0', session='manual-test')
