"""Small, revision-checked simulation protocol; never retries a mutation."""
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler

import requests

PROTOCOL = 'chess_teacher_state_v1'


class StateStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot = None
        self.commands = queue.Queue(maxsize=1)

    def publish(self, observation, sequence, timing=None, reset_context=None):
        snapshot = dict(protocol=PROTOCOL, sequence=sequence,
                        simulation_time=observation['physics_time'], observation=observation,
                        timing_ms=timing or {}, camera_render_products=0)
        if reset_context is not None:
            snapshot['reset_context']=reset_context
        with self.lock:
            self.snapshot = snapshot
        return snapshot


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Applied commands are never silently retried.

        def do_GET(self):
            if self.path != '/state':
                self.respond({'error': 'Use /state; this server does not render cameras'}, 404)
                return
            with store.lock:
                snapshot = store.snapshot
            self.respond(snapshot or {'error': 'Starting'}, 200 if snapshot else 503)

        def do_POST(self):
            if self.path not in ('/reset', '/step', '/save_reset', '/restore_reset'):
                self.respond({'error': 'Unknown command'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected a small JSON body')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or type(payload.get('sequence')) is not int:
                    raise ValueError('sequence is required')
                command = dict(path=self.path, payload=payload, done=threading.Event(),
                               expires=time.monotonic()+30)
                store.commands.put_nowait(command)
            except (ValueError, queue.Full) as error:
                self.respond({'error': str(error)}, 400)
                return
            if not command['done'].wait(30):
                self.respond({'error': 'Timeout; execution uncertain. Read state and reset.'}, 504)
                return
            self.respond(command['result'], command.get('status', 200))
    return Handler


class StateClient:
    def __init__(self, url='http://127.0.0.1:8081'):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.session.trust_env = False  # Local simulation does not use HTTP proxies.
        self.latest = None
        self.uncertain = True

    def read(self):
        response = self.session.get(self.url+'/state', timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get('protocol') != PROTOCOL:
            raise RuntimeError('Not a state-only teacher server')
        self.latest = result
        return result

    def _command(self, path, **values):
        if self.latest is None:
            self.read()
        before = self.latest
        self.uncertain = True
        response = self.session.post(self.url+path, json={'sequence': before['sequence'], **values}, timeout=35)
        response.raise_for_status()
        result = response.json()
        if result.get('protocol') != PROTOCOL or result['sequence'] != before['sequence']+1:
            raise RuntimeError('Concurrent controller or mismatched state revision')
        if path == '/step':
            delta = result['simulation_time']-before['simulation_time']
            if abs(delta-1/30) > 1e-6:
                raise RuntimeError('Unexpected simulated step duration')
        self.latest, self.uncertain = result, False
        return result

    def reset(self, options):
        self.read()  # Recover current revision after an uncertain command.
        return self._command('/reset', options=options)

    def step(self, action):
        if self.uncertain:
            raise RuntimeError('reset() required after uncertain execution')
        return self._command('/step', action=list(map(float, action)))

    def save_reset(self,name,provenance):
        if self.uncertain:
            raise RuntimeError('reset() required after uncertain execution')
        return self._command('/save_reset',name=name,provenance=provenance)

    def restore_reset(self,name):
        self.read()
        return self._command('/restore_reset',name=name)

    def close(self):
        self.session.close()
