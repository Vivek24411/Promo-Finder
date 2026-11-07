"""A fake OpenAI-compatible server for tests and benchmarks - no network, no cost.

Behaviour is chosen by the MODEL NAME, so any test can ask for a failure mode:
    mock                 well-formed JSON
    mock-fenced          ```json fenced```            mock-think     <think>…</think> before the JSON
    mock-trailing        JSON + commentary with {braces}    mock-badids    adds invented beat IDs
    mock-429x2           first 2 calls answer 429     mock-503x1      first call answers 503
    mock-timeout         every call sleeps past the client timeout
    mock-latency=0.3     each call takes 0.3 s (any mode can add it, e.g. mock-fenced-latency=0.2)
Replies are plausible: the miner picks real beat IDs from the prompt, the judge scores every CLIP,
the profile returns words that occur in the prompt, the reranker returns a full permutation.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOOKY = re.compile(r"slap|throw|wine|gun|scream|fired|wife|pregnant|alive|crash|grab|divorce|tears", re.I)


def _reply(user: str, rnd: random.Random) -> dict:
    if "Order ALL" in user:
        labels = re.findall(r"^CLIP ([A-Z]) \(", user, re.M)
        rnd.shuffle(labels)
        return {"order": labels}
    if "CLIP 1" in user:
        n = len(re.findall(r"^CLIP \d+", user, re.M))
        return {"clips": [{"clip": i + 1, "hook": rnd.randint(3, 9), "cliffhanger": rnd.randint(4, 9),
                           "standalone": rnd.randint(2, 9), "hook_moment": "mock hook", "open_question": "mock question",
                           "confusing": "" if i % 3 else "who is she?", "why": "mock why",
                           "unanswered": rnd.random() > 0.3, "stakes": rnd.random() > 0.4,
                           "resolved": rnd.random() > 0.7} for i in range(n)]}
    if "Describe this show" in user:
        words = sorted(set(w.lower() for w in re.findall(r"[A-Za-z]{6,}", user)))[:30]
        return {"genre": "mock drama", "characters": [{"name": "MOCK", "role": "protagonist", "relationships": "-"}],
                "reveal_words": words[:6] + ["neverappearsanywhere"], "threat_words": words[6:10]}
    beats = re.findall(r"^\[(e\d+b\d+) [^\]]*\] (.*)$", user, re.M)
    hooky = [b for b, t in beats if HOOKY.search(t)] or [b for b, _ in beats]
    pick = rnd.sample(hooky, min(4, len(hooky)))
    tail = [b for b, _ in beats[-6:]]
    return {"hooks": [{"beat": b, "type": "emotional_reveal" if i == 0 else "reveal", "strength": rnd.randint(5, 9),
                       "why": "mock"} for i, b in enumerate(pick)],
            "cliffhangers": [{"beat": b, "strength": rnd.randint(5, 9), "open_question": "?"}
                             for b in rnd.sample(tail, min(3, len(tail)))]}


class MockLLM:
    def __init__(self):
        self.calls = 0
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()
        self.bodies: list[dict] = []
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model = body.get("model", "mock")
                with srv.lock:
                    srv.calls += 1
                    srv.counts[model] = srv.counts.get(model, 0) + 1
                    nth = srv.counts[model]
                    srv.bodies.append({"headers": dict(self.headers), "body": body})
                if m := re.search(r"latency=([\d.]+)", model):
                    time.sleep(float(m.group(1)))
                for code in (429, 503):
                    if m := re.search(rf"{code}x(\d+)", model):
                        if nth <= int(m.group(1)):
                            self.send_response(code)
                            self.send_header("Retry-After", "0")
                            self.end_headers()
                            self.wfile.write(b'{"error":"busy"}')
                            return
                if "timeout" in model:
                    time.sleep(3)
                user = body["messages"][-1]["content"]
                data = _reply(user, random.Random(len(user)))
                if "badids" in model and "hooks" in data:
                    data["hooks"].append({"beat": "e99b999", "strength": 9})
                text = json.dumps(data)
                if "fenced" in model:
                    text = "Here you go:\n```json\n" + text + "\n```"
                if "think" in model:
                    text = "<think>The user wants {json}. Let me consider {a: b}...</think>\n" + text
                if "trailing" in model:
                    text = text + "\n\nNote: I scored these {harshly} as asked."
                out = json.dumps({"choices": [{"message": {"content": text}}],
                                  "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 150}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    with MockLLM() as m:
        print(f"mock LLM listening on {m.url}  (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
