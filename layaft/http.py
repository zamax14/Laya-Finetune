"""The one HTTP client every LLM, backend and teacher uses: stdlib only."""
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(url, body, key=None, timeout=180):
    """POST JSON; one retry if the server fails (5xx), 4xx errors surface as they are."""
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body, ensure_ascii=False).encode()
    for attempt in range(2):
        try:
            with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code < 500 or attempt:
                try:
                    detail = json.load(exc)["error"]["message"]
                except Exception:
                    detail = exc.reason
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
            time.sleep(1)
