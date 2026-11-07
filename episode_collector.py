"""Lossless simulation episode bundles for subsequent ACT dataset conversion.

Each frame contains the before-action camera pair and legacy state, plus the
actual executed action. Only verified complete-move successes are accepted.
"""
import json
from pathlib import Path
import numpy as np


class EpisodeCollector:
    def __init__(self, directory, initial, provenance):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.initial = initial
        self.provenance = provenance
        self.frames = 0
        self.timestamps = []
        self.actions = []
        self.states = []
        self.finalized = False
        (self.directory / 'initial.json').write_text(json.dumps(initial, indent=2))
        (self.directory / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    def write(self, before_images, before_metadata, info):
        if self.finalized:
            raise RuntimeError('Episode has already been finalized')
        state = np.asarray(before_images['state'], np.float32)
        executed = np.asarray(info['residual']['executed_legacy'], np.float32)
        timestamp = float(before_metadata['simulation_time'])
        if (state.shape != (6,) or executed.shape != (6,) or not np.isfinite(state).all()
                or not np.isfinite(executed).all() or not np.isfinite(timestamp)):
            raise ValueError('Invalid state, executed action or timestamp')
        if self.timestamps and not np.isclose(timestamp - self.timestamps[-1], 1 / 30, atol=1e-6):
            raise ValueError('Collection lost a simulated frame')
        if info['telemetry']['frame_id'] != before_metadata['frame_id'] + 1:
            raise ValueError('Action/observation frame mismatch')
        np.testing.assert_allclose(state, before_metadata['observation']['observation_state'], atol=1e-5)
        if not np.isclose(info['telemetry']['simulation_time'] - timestamp, 1 / 30, atol=1e-6):
            raise ValueError('Executed action does not have the expected timestep')
        images = {key: np.array(before_images[key], copy=True) for key in ('laptop_rgb', 'phone_rgb')}
        for image in images.values():
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError('Expected two lossless RGB camera images at 640x480')
        np.savez_compressed(self.directory / f'frame_{self.frames:06d}.npz', **images,
                            observation_state=state, action_legacy=executed,
                            timestamp=np.float64(timestamp), frame_id=np.int64(before_metadata['frame_id']),
                            next_frame_id=np.int64(info['telemetry']['frame_id']))
        self.timestamps.append(timestamp)
        self.states.append(state.copy())
        self.actions.append(executed.copy())
        self.frames += 1

    def finalize(self, score, error=None):
        if self.finalized:
            raise RuntimeError('Episode has already been finalized')
        self.finalized = True
        # Pickup diagnostic successes do not certify a complete chess move.
        geometry = self.initial['telemetry']['observation'].get('task_geometry', {})
        capture = score.get('capture') or {}
        capture_ok = geometry.get('capture_index') is None or (capture.get('released_stable', False) and capture.get('upright_stable', False))
        accepted = bool(self.frames and not error and score.get('success') and
                        score.get('objective') != 'capture_pickup_only' and
                        score.get('reason') == 'success' and score.get('other_pieces_disturbed', 0) == 0
                        and score.get('placement_streak', 0) >= 10 and score.get('home_streak', 0) >= 10
                        and score.get('lifted', False) and capture_ok)
        np.savez_compressed(self.directory / 'trajectory.npz',
                            timestamps=np.array(self.timestamps, np.float64),
                            states=np.array(self.states, np.float32), actions=np.array(self.actions, np.float32))
        manifest = {'format': 'chess_sim_lossless_episode_v1', 'accepted_for_act_training': accepted,
                    'frames': self.frames, 'fps': 30, 'score': score, 'error': error,
                    'image_order': 'RGB', 'state_and_action_units': 'source legacy calibration, unchanged',
                    'action_source': 'actual simulator executed target after residual, timing and physical clipping',
                    'source': 'automatically scored simulation; not human expert demonstration',
                    'capture_collection_quality': 'victim released stably and upright; non-upright captures remain valid moves but are excluded from ACT training',
                    'acceptance_version':'complete_upright_move_v2',
                    'transported_flag_role':'diagnostic reward event, not an additional completion requirement',
                    'provenance': self.provenance}
        (self.directory / 'episode.json').write_text(json.dumps(manifest, indent=2))
        return manifest
