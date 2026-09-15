"""Small live integration check; requires updated --control --contacts server."""
import numpy as np

from chess_env import ChessEnv, ResidualACTEnv
from act_client import command


def main():
    env = ChessEnv(max_steps=2, pawn_jitter_mm=.5)
    obs, first = env.reset(seed=23)
    assert env.observation_space.contains(obs)
    assert set(obs) == {'state', 'laptop_rgb', 'phone_rgb'}
    frame = env.latest['frame_id']
    for options in [{'pawn_offset_mm': [3, 0]}, {'pawn_offset_mm': {'x': 0}}, {'unknown': 1}]:
        try:
            command(env.url, '/reset', frame, options=options)
        except RuntimeError as error:
            assert 'Simulator rejected' in str(error)
        else:
            raise AssertionError('Invalid reset options accepted')
    assert env.client.read()[0]['frame_id'] == frame
    for bad in [np.zeros(5), np.full(6, np.nan), np.full(6, 1e6)]:
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action accepted')
    assert env.client.read()[0]['frame_id'] == frame
    _, reward, terminated, truncated, info = env.step(obs['state'])
    assert not terminated and not truncated and reward <= 0
    _, _, terminated, truncated, _ = env.step(obs['state'])
    assert truncated and not terminated
    try:
        env.step(obs['state'])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Post-terminal step accepted')
    obs, second = env.reset(seed=23)
    assert first['reset_options'] == second['reset_options']
    np.testing.assert_allclose(first['telemetry']['observation']['pieces']['pawn_center_m'],
                               second['telemetry']['observation']['pieces']['pawn_center_m'], atol=.0001)
    target = second['telemetry']['observation']['task_geometry']['target_center_m']
    _, shifted = env.reset(options={'pawn_offset_mm': [1, -1]})
    np.testing.assert_equal(target, shifted['telemetry']['observation']['task_geometry']['target_center_m'])

    class HoldPolicy:
        config = {'n_action_steps': 100}
        def reset(self):
            pass
        def action(self, state, *_):
            return state.copy()

    residual = ResidualACTEnv(env, HoldPolicy())
    obs, _ = residual.reset(seed=23)
    assert residual.observation_space.contains(obs)
    before = obs['base_action'].copy()
    obs, _, _, _, info = residual.step(np.zeros(6, np.float32))
    np.testing.assert_allclose(info['residual']['executed_legacy'], before)
    assert not info['residual']['limited']
    assert residual.observation_space.contains(obs)
    residual.reset(options={'pawn_offset_mm': [0, 0]})
    residual.close()
    print('PASS: spaces, action validation, seeded offsets, fixed target, 30 Hz stepping, timeout, reset, zero residual')


if __name__ == '__main__':
    main()
