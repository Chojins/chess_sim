"""Integration checks against a running camera server with --control."""
import time
import numpy as np
from act_client import command
from camera_client import PairedCameraClient


def main():
    url = 'http://127.0.0.1:8080'
    client = PairedCameraClient(url)
    meta, images = client.read()
    command(url, '/reset', meta['frame_id'])
    before, images = client.read()
    assert set(images) == {'blue', 'red'}
    assert all(image.shape == (480, 640, 3) for image in images.values())
    for invalid, expected in [(np.full(6, 1e6), 'Out-of-limit'), (np.full(6, np.nan), 'finite')]:
        try:
            command(url, '/step', before['frame_id'], invalid)
        except RuntimeError as error:
            assert expected in str(error), error
        else:
            raise AssertionError('Invalid action was accepted')
    held, _ = client.read()
    assert held['frame_id'] == before['frame_id']
    response = command(url, '/step', before['frame_id'], before['observation']['observation_state'])
    after, _ = client.read()
    assert response['frame_id'] == after['frame_id'] == before['frame_id'] + 1
    assert after['observation']['control_step'] == 1
    np.testing.assert_allclose(after['simulation_time'] - before['simulation_time'], 1 / 30, atol=1e-6)
    try:
        command(url, '/step', before['frame_id'], before['observation']['observation_state'])
    except RuntimeError as error:
        assert 'Stale' in str(error)
    else:
        raise AssertionError('Stale action was accepted')
    time.sleep(0.5)
    idle, _ = client.read()
    assert idle['frame_id'] == after['frame_id']
    assert idle['simulation_time'] == after['simulation_time']
    command(url, '/reset', idle['frame_id'])
    reset, _ = client.read()
    assert reset['observation']['episode_id'] == before['observation']['episode_id'] + 1
    assert reset['observation']['control_step'] == 0
    np.testing.assert_allclose(reset['observation']['observation_state'], before['observation']['observation_state'], atol=0.05)
    if 'pieces' in reset['observation']:
        np.testing.assert_allclose(reset['observation']['pieces']['piece_centers_m'],
                                   before['observation']['pieces']['piece_centers_m'], atol=0.0001)
    print('PASS: paired observations, finite/limit/stale rejection, 30 Hz step, idle pause, robot and piece reset')


if __name__ == '__main__':
    main()
