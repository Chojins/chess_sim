"""Optional state-teacher BC initialization from audited v29 traces.

This is an initial motion prior, including the known pauses/home detour. It is
not a new expert, a replay buffer, or evidence of improved movement quality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from chess_task import LegalMoveTask, TaskConfig
from teacher.environment import TeacherFeatures


def load_prior(directories, specification, anchors, config):
    if len(directories) < 2:
        raise ValueError('Need separate whole training and validation episodes')
    observations, actions, validation, episodes, sources = [], [], [], [], []
    identity = set()
    for episode, directory in enumerate(map(Path, directories)):
        from verify_collected_episodes import audit_episode
        # Audit the physical episode, not just its saved success flag.
        bundle = directory/'lossless_episode'
        audit_episode(bundle)
        initial = json.loads((directory/'initial.json').read_text())['telemetry']['observation']
        trace = directory/'steps.jsonl'
        digest = hashlib.sha256(trace.read_bytes()).hexdigest()
        if digest in identity:
            raise ValueError('Duplicate source episode would leak the validation split')
        identity.add(digest)
        task = LegalMoveTask(initial, TaskConfig(max_steps=config.max_steps, capture_home_retry=True))
        features = TeacherFeatures(initial, specification, anchors, config)
        actual = initial
        mapping = initial['joint_map']
        scale, sim, legacy_home = [np.asarray(mapping[key]) for key in ('scale', 'sim_home_deg', 'legacy_home')]
        command = sim+scale*(np.asarray(initial['target_legacy_action'])-legacy_home)
        velocity = np.zeros(6)
        previous_q = np.asarray(actual['measured_joint_deg'])
        limits = np.array(initial['joint_limits_deg'])
        low, high = limits[:, 0]+1e-4, limits[:, 1]-1e-4
        command = np.clip(command, low, high)
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        limited = 0
        for index, row in enumerate(rows):
            target = sim+scale*(np.asarray(row['residual']['executed_legacy'])-legacy_home)
            target = np.clip(target, low, high)
            observations.append(features.encode(actual, task, command, velocity, previous_q))
            actions.append((2*(target-low)/(high-low)-1).astype(np.float32))
            validation.append(episode == len(directories)-1)
            episodes.append(episode)
            limited += int(np.any(np.abs(target-command)*30 > np.asarray(config.max_velocity_deg_s)+1e-5))
            previous_q = np.asarray(actual['measured_joint_deg'])
            velocity = (target-command)*30
            command = target
            actual = {**actual, **row['placement_telemetry']}
            _, terminated, truncated, score = task.update(actual)
            if (terminated or truncated) and index != len(rows)-1:
                raise ValueError('Source scoring terminates before its trace ends')
        if not score['success']:
            raise ValueError('Source trace does not independently complete the move')
        sources.append({'directory': str(directory.resolve()), 'trace_sha256': digest,
                        'frames': len(rows), 'validation': episode == len(directories)-1,
                        'commands_exceeding_new_velocity_limit': limited})
    data = dict(observations=np.asarray(observations, np.float32), actions=np.asarray(actions, np.float32),
                validation=np.asarray(validation, bool), episode=np.asarray(episodes, np.int32))
    return data, {'sources': sources, 'known_motion_defects': 'pauses and intermediate home return',
                  'purpose': 'BC initialization only; never imported into the revised dynamics/reward replay',
                  'validation_split': 'last whole source episode; excluded from all gradient updates'}
