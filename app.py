"""
app.py - Flask backend for the SUMO multi-agent traffic dashboard.

One background thread owns the (single, global) TraCI connection and steps the
shared simulation using the trained PPO models; Flask threads only read
snapshots. If no trained models are found it falls back to a random "demo"
policy so the dashboard always works.

Run:  python app.py --config simulation/config.sumocfg --models-dir ./models
Open: http://127.0.0.1:5000
"""
import argparse
import atexit
import collections
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402

from rl.environment import AGENT_IDS, MultiAgentTrafficEnv  # noqa: E402

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

app = Flask(__name__)
runner = None


class SimRunner(threading.Thread):
    def __init__(self, config, models_dir, suffix, max_steps):
        super().__init__(daemon=True)
        self.env = MultiAgentTrafficEnv(sumo_config=config, use_gui=False, max_steps=max_steps)
        self.models = self._load(models_dir, suffix)
        self.lock = threading.Lock()
        self.go = threading.Event()
        self.go.set()                     # auto-start
        self.reset_req = False
        self.delay = 0.15                 # seconds between sim steps
        self.deterministic = True
        self.episode = 0
        self.step_no = 0
        self.network = None
        self.snapshot = None
        self.error = None
        self.events = collections.deque(maxlen=40)
        self.event_id = 0
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _load(models_dir, suffix):
        if PPO is None:
            return {}
        models = {}
        for aid in AGENT_IDS:
            try:
                models[aid] = PPO.load(f"{models_dir}/ppo_{aid}{suffix}")
            except Exception as exc:
                print(f"[dashboard] no model for {aid} ({exc}) -> demo mode")
                return {}
        print("[dashboard] trained PPO models loaded")
        return models

    def _log(self, text, kind="info", agent=None):
        self.event_id += 1
        self.events.appendleft({"id": self.event_id, "step": self.step_no,
                                "text": text, "kind": kind, "agent": agent})

    @staticmethod
    def _read_network():
        lanes = []
        for lid in traci.lane.getIDList():
            shape = traci.lane.getShape(lid)
            lanes.append({"p": [[round(x, 1), round(y, 1)] for x, y in shape],
                          "i": lid.startswith(":")})
        return {"lanes": lanes}

    # ----------------------------------------------------------- simulation
    def _new_episode(self):
        self.reset_req = False
        self.obs, _ = self.env.reset()
        self.episode += 1
        self.step_no = 0
        self.stats = {a: dict(reward=0.0, arrivals=0, collisions=0, steps=0, speed_sum=0.0)
                      for a in AGENT_IDS}
        self.actions = {a: 2 for a in AGENT_IDS}
        self.rewards = {a: 0.0 for a in AGENT_IDS}
        if self.network is None:
            self.network = self._read_network()
        self._log(f"Episode {self.episode} started")
        self._publish()

    def _act(self, aid):
        if self.models:
            return int(self.models[aid].predict(self.obs[aid], deterministic=self.deterministic)[0])
        return int(self.rng.choice([0, 2, 2, 4, 3, 0]))

    def _tick(self):
        acts = {a: self._act(a) for a in self.env.agents}
        obs, rew, _term, _trunc, info = self.env.step(acts)
        self.step_no += 1
        self.obs, self.actions, self.rewards = obs, acts, rew
        for a in AGENT_IDS:
            s = self.stats[a]
            s["reward"] += rew.get(a, 0.0)
            s["steps"] += 1
            s["speed_sum"] += float(obs[a][0]) if a in obs else 0.0
            ev = info.get(a, {}).get("event")
            if ev == "arrived":
                s["arrivals"] += 1
                self._log(f"{a} reached its destination", "good", a)
            elif ev == "collision":
                s["collisions"] += 1
                self._log(f"{a} collided", "bad", a)
        self._publish()
        if not self.env.agents:           # joint horizon reached -> restart
            self._new_episode()

    def _publish(self):
        present = set(traci.vehicle.getIDList())
        vehicles = []
        for v in present:
            if v not in AGENT_IDS:
                x, y = traci.vehicle.getPosition(v)
                vehicles.append([round(x, 1), round(y, 1), round(traci.vehicle.getAngle(v))])
        agents = {}
        for a in AGENT_IDS:
            o, s = self.obs.get(a), self.stats[a]
            d = {"present": a in present, "action": self.actions.get(a, 2),
                 "reward": round(float(self.rewards.get(a, 0.0)), 2),
                 "total": round(s["reward"], 1), "arrivals": s["arrivals"],
                 "collisions": s["collisions"],
                 "avg_speed": s["speed_sum"] / max(s["steps"], 1)}
            if o is not None:
                d.update(speed=float(o[0]), leader_speed=float(o[1]), gap=float(o[2]),
                         wait=float(o[3]), accel=float(o[4]), lane=int(o[5]),
                         lanes=int(o[6]), density=int(o[7]), tls=int(o[8]),
                         tls_dist=float(o[9]), progress=float(o[11]))
            if a in present:
                x, y = traci.vehicle.getPosition(a)
                d.update(x=round(x, 1), y=round(y, 1), a=traci.vehicle.getAngle(a),
                         edge=traci.vehicle.getRoadID(a))
            agents[a] = d
        snap = {"ready": True, "step": self.step_no, "episode": self.episode,
                "running": self.go.is_set(), "mode": "trained PPO" if self.models else "demo (random)",
                "deterministic": self.deterministic, "agents": agents,
                "vehicles": vehicles, "events": list(self.events)}
        with self.lock:
            self.snapshot = snap

    def run(self):
        try:
            self._new_episode()
            while True:
                if self.reset_req:
                    self._new_episode()
                    continue
                if not self.go.wait(0.2):
                    continue
                self._tick()
                time.sleep(self.delay)
        except Exception as exc:          # surface the problem in the UI
            self.error = repr(exc)
            print("[dashboard] simulation error:", exc)


# ------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/network")
def api_network():
    if runner.network is None:
        return jsonify(error="not ready"), 503
    return jsonify(runner.network)


@app.route("/api/state")
def api_state():
    with runner.lock:
        snap = dict(runner.snapshot or {"ready": False})
    snap["running"] = runner.go.is_set()
    snap["error"] = runner.error
    return jsonify(snap)


@app.post("/api/control")
def api_control():
    d = request.get_json(force=True, silent=True) or {}
    cmd = d.get("cmd")
    if cmd == "start":
        runner.go.set()
    elif cmd == "pause":
        runner.go.clear()
    elif cmd == "reset":
        runner.reset_req = True
    if "speed" in d:
        m = float(d["speed"])
        runner.delay = 0.0 if m <= 0 else 0.15 / m
    if "deterministic" in d:
        runner.deterministic = bool(d["deterministic"])
    return jsonify(ok=True)


def main():
    global runner
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="simulation/config.sumocfg")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--suffix", default="_final")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    runner = SimRunner(a.config, a.models_dir, a.suffix, a.max_steps)
    runner.start()
    atexit.register(runner.env.close)
    app.run(host=a.host, port=a.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()