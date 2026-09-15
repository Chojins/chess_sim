"""Headless, real PhysX checks of centring, lift-off attenuation and replay.

Run using Isaac's python.sh from this directory. Results go to --output.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/magnets/verification'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False})
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdRender
        from board_magnets import MagnetSettings, attraction
        from episode_replay import PhysicsEpisodeReplay
        context = omni.usd.get_context()
        assert context.open_stage(str(Path('Sim_Stage.usd').resolve()))
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdRender.Product):
                prim.SetActive(False)
        driver = PhysicsEpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
            'calibration/legacy_blue_joint_map_physics.json', args.output,
            contacts=True, speed=0.5, magnets=MagnetSettings())
        driver.external_control = True
        magnets, pieces = driver.magnets, driver.pieces
        pawn = pieces.paths.index('/World/chess_pieces_blue/pawn_03')
        desired = pieces.initial_centers[pawn] + [0, 0.045, 0]
        square = magnets.squares[np.argmin(np.linalg.norm(magnets.squares[:, :2] - desired[:2], axis=1))]
        results = {'settings': magnets.describe(), 'centring': []}

        def reset():
            pieces.reset()
            for color, robot in driver.robots.items():
                q = driver.initial_robot_poses[color]
                robot.set_joint_positions(q)
                robot.set_joint_velocities(np.zeros_like(q))
                robot.set_joint_position_targets(q)

        # The disabled case uses identical low board friction: any centring
        # must come from magnetic forces, not a sloping board or reset logic.
        for enabled, offset in [(False, [8, 0]), (True, [8, 0]), (True, [-8, 0]),
                                (True, [0, 10]), (True, [6, 6])]:
            reset()
            driver.magnets = magnets if enabled else None
            positions, rotations = pieces.view.get_world_poses()
            base = magnets.base_positions()[pawn]
            start = square + [offset[0] / 1000, offset[1] / 1000, 0.0001]
            positions[pawn] += start - base
            pieces.view.set_world_poses(positions, rotations)
            trajectory = []
            for _ in range(60):  # Four seconds of simulated time.
                driver.advance(driver.targets[0])
                trajectory.append(magnets.base_positions()[pawn].tolist())
            end = magnets.base_positions()[pawn]
            error = float(np.linalg.norm(end[:2] - square[:2]))
            sample = pieces.sample()
            result = {'enabled': enabled, 'offset_mm': offset, 'final_error_m': error,
                      'upright_cosine': sample['pawn_upright_cosine'], 'trajectory_m': trajectory,
                      'passed': bool(error < 0.0015 and sample['pawn_upright_cosine'] > .99) if enabled
                      else bool(abs(error - np.linalg.norm(offset) / 1000) < .0005)}
            results['centring'].append(result)
            print('CENTRING', json.dumps({k: v for k, v in result.items() if k != 'trajectory_m'}), flush=True)
        points = np.tile(square, (2, 1))
        points[1, 2] += .02
        force = attraction(points, np.zeros_like(points), magnets.squares, magnets.settings)
        results['lift_force_ratio_at_20mm'] = float(np.linalg.norm(force[1]) / np.linalg.norm(force[0]))
        reset()
        driver.magnets = magnets
        driver.external_control = False
        driver.index = -1
        driver.tracking.clear()
        for _ in driver.targets:
            replay = driver.step()
        sample = replay['pieces']
        results['replay'] = sample
        results['replay_square_xy_error_m'] = float(np.linalg.norm(np.array(sample['pawn_center_m'])[:2] - square[:2]))
        results['replay_passed'] = bool(sample['pawn_lift_max_m'] > .005 and
            sample['pawn_target_xy_error_m'] < .005 and sample['pawn_upright_cosine'] > .95 and
            abs(sample['pawn_height_change_m']) < .003 and sample['other_pieces_displaced_over_3mm'] == 0)
        results['passed'] = bool(all(x['passed'] for x in results['centring']) and
                                  results['lift_force_ratio_at_20mm'] < .01 and results['replay_passed'])
        (args.output / 'magnet_report.json').write_text(json.dumps(results, indent=2))
        print('MAGNET_VERIFICATION', json.dumps({k: v for k, v in results.items()
              if k not in ('settings', 'centring', 'replay')}), flush=True)
        if not results['passed']:
            raise RuntimeError('Magnetic physics verification failed; see magnet_report.json')
    except Exception as error:
        import traceback
        traceback.print_exc()
        failure = locals().get('results', {})
        failure.update(passed=False, error=str(error))
        (args.output / 'magnet_report.json').write_text(json.dumps(failure, indent=2))
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
