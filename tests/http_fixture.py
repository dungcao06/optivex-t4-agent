"""Local protocol simulator, never an accuracy or production-House-model test.

Replies come only from the prompt using the pinned organizer's public mock.
The HTTP transport is real so CLI subprocesses exercise serialization, headers,
URL construction, reply parsing, retries, and output writing end to end.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
sys.path.insert(0, str(UPSTREAM))

from baselines.strong_rag_baseline.cli import _mock_reply  # noqa: E402

BATCH_MARKER = "BATCH REQUESTS JSON:\n"
UNSUPPORTED_CLAIM = "UNSUPPORTED SENTINEL: the evidence guarantees a 999 percent return."


def prompt_reply(payload: dict, mode: str) -> str:
    messages = payload["messages"]
    system = next(message["content"] for message in messages if message["role"] == "system")
    user = next(message["content"] for message in reversed(messages) if message["role"] == "user")

    if mode == "review_valid" and user.startswith("ROSTER REVIEW REQUEST."):
        entity_id = re.search(r"^- ([^:]+): point_forecast=", user, re.M).group(1)
        return json.dumps({"updates": [{"entity_id": entity_id, "point_forecast": 0.5}]})

    def reply_one(prompt: str) -> dict:
        reply = json.loads(_mock_reply(system, prompt))
        if mode == "ungrounded":
            for evidence in reply["evidence"]:
                evidence["quote"] = "The revenue earnings rose by 999 percent without any risk."
                evidence["claim"] = UNSUPPORTED_CLAIM
        elif mode == "invalid_rank":
            reply["rank"] = -3
        elif mode == "nonfinite":
            reply["point_forecast"] = float("nan")
            reply["interval"] = {"level": 0.9, "lo": -float("inf"), "hi": float("inf")}
        elif mode == "interval_list":
            reply["interval"] = [-1, 1]
        elif mode == "doc_id_list":
            for evidence in reply["evidence"]:
                evidence["doc_id"] = [evidence["doc_id"]]
        return reply

    if BATCH_MARKER in user:
        requests = json.loads(user.split(BATCH_MARKER, 1)[1])
        result = {"predictions": [dict(reply_one(item["prompt"]), entity_id=item["entity_id"]) for item in requests]}
    else:
        result = reply_one(user)
    raw = json.dumps(result)
    if mode == "thinking":
        return '<think>Internal scratch work includes {"candidate": "wrong"}.</think>\n```json\n' + raw + "\n```"
    return raw


@contextmanager
def model_server(mode: str = "valid"):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(raw)
            requests.append({"path": self.path, "authorization": self.headers.get("Authorization"), "payload": payload})
            if mode == "unavailable" or (mode == "transient_http" and len(requests) == 1):
                body = b'{"error": "fixture temporarily unavailable"}'
                self.send_response(503)
            elif mode == "invalid_envelope":
                body = b'{"choices": [null]}'
                self.send_response(200)
            elif mode == "truncated_envelope":
                body = b'{"choices": ['
                self.send_response(200)
                self.close_connection = True
            else:
                text = "This is not JSON" if mode == "malformed_first" and len(requests) == 1 else prompt_reply(payload, mode)
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + (100 if mode == "truncated_envelope" else 0)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
