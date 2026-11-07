"""Measure real state-only physics throughput while holding the home command."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from teacher.environment import CASE
from teacher.transport import StateClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    client = StateClient(args.url)
    try:
        first = client.reset({'history': CASE['history'], 'move': CASE['move']})
        action = first['observation']['target_legacy_action']
        timings = []
        started = time.perf_counter()
        for _ in range(args.steps):
            before = time.perf_counter()
            latest = client.step(action)
            timings.append((time.perf_counter()-before)*1000)
        elapsed = time.perf_counter()-started
        simulated = latest['simulation_time']-first['simulation_time']
        if not np.isclose(simulated, args.steps/30, atol=1e-5):
            raise RuntimeError('Lost or duplicated physics transition')
        report = {'steps': args.steps, 'wall_seconds': elapsed, 'simulated_seconds': simulated,
                  'steps_per_wall_second': args.steps/elapsed, 'real_time_factor': simulated/elapsed,
                  'median_step_ms': float(np.median(timings)), 'p95_step_ms': float(np.percentile(timings, 95)),
                  'camera_render_products': latest['camera_render_products'],
                  'parallel_environments': 1, 'workload': 'hold home command; not a grasp success test',
                  'physics_settings': latest['observation']['physics_settings']}
        (args.output/'benchmark.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != 'physics_settings'}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
