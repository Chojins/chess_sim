"""Run the existing chess physics without camera products or ACT inference.

Run using teacher/run_server.sh. All Isaac access stays on the main thread.
"""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
from http.server import ThreadingHTTPServer

from teacher.transport import StateStore, handler_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--usd', type=Path, default=Path('Sim_Stage.usd'))
    parser.add_argument('--episode', type=Path, default=Path('outputs/replay/episode_000000.npz'))
    parser.add_argument('--joint-map', type=Path, default=Path('calibration/legacy_blue_joint_map_physics.json'))
    parser.add_argument('--reset-bank',type=Path,help='Reuse a compatible checkpoint bank; defaults to OUTPUT/reset_bank')
    parser.add_argument('--render-cameras',action='store_true',help='Review only: attach both authored cameras and expose synchronized /frames.npz')
    parser.add_argument('--resolution',type=int,nargs=2,default=[640,480])
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be within 1..65535')
    if min(args.resolution)<1: parser.error('Camera resolution must be positive')
    for path in (args.usd, args.episode, args.joint_map):
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app_config={'headless': True, 'hide_ui': True,'disable_viewport_updates': True, 'multi_gpu': False}
    if args.render_cameras:
        app_config.update(renderer='RaytracedLighting',anti_aliasing=2,width=args.resolution[0],height=args.resolution[1])
    app = SimulationApp(app_config)
    server = None
    camera_capture=None
    try:
        import omni.usd
        import numpy as np
        from pxr import UsdRender
        from board_magnets import MagnetSettings
        from sim_control import SimControl
        from teacher.reset_bank import ResetBank

        context = omni.usd.get_context()
        if not context.open_stage(str(args.usd.resolve())):
            raise RuntimeError('Could not open chess stage')
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        # Disable every saved render product, not just those named Replicator.
        products = [prim for prim in stage.Traverse() if prim.IsA(UsdRender.Product)]
        disabled = [str(prim.GetPath()) for prim in products]
        for prim in products:
            prim.SetActive(False)
        for prim in stage.Traverse():
            if prim.IsA(UsdRender.Settings):
                UsdRender.Settings(prim).GetProductsRel().SetTargets([])
        control = SimControl(stage, args.episode, args.joint_map, args.output,
                             contacts=True, magnets=MagnetSettings(energy_j=.00056, board_friction=.12))
        # Match the baseline's 240 Hz physics and 30 Hz control contract.
        if not np.isclose(control.driver.fps, 30):
            raise RuntimeError('Teacher currently requires the 30 Hz source episode')
        bank=ResetBank(control,args.reset_bank or args.output/'reset_bank',
                       [layer.realPath for layer in stage.GetUsedLayers() if layer.realPath])
        def observe_extra(observation):
            robot=control.driver.robots['blue']
            observation['joint_velocity_deg_s']=np.rad2deg(robot.get_joint_velocities()[0][control.driver.indices]).tolist()
            observation['piece_velocities_m_s_rad_s']=control.driver.pieces.view.get_velocities().tolist()
            return observation
        manifest = dict(mode='state_teacher_camera_review' if args.render_cameras else 'state_only_teacher',
                        camera_render_products=2 if args.render_cameras else 0,
                        disabled_saved_products=disabled, usd=str(args.usd.resolve()),
                        joint_map=str(args.joint_map.resolve()), physics_hz=240, control_hz=30,
                        contacts=True, magnets=True, parallel_environments=1,
                        note=('Existing CPU physics backend; paired cameras for frozen-policy review'
                              if args.render_cameras else 'Existing CPU physics backend; no Replicator or policy camera rendering'))
        (args.output/'server_manifest.json').write_text(json.dumps(manifest, indent=2))
        if args.render_cameras:
            from teacher.rendering import CameraCapture,CameraStore,camera_handler_for
            camera_capture=CameraCapture(stage,control,args.resolution)
            store=CameraStore(camera_capture);handler=camera_handler_for(store)
        else:
            store=StateStore();handler=handler_for(store)
        sequence=0
        store.publish(observe_extra(control.observe()), sequence)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print('TEACHER_STATE_READY', json.dumps(manifest), flush=True)
        fault = False
        while app.is_running():
            try:
                command = store.commands.get(timeout=.1)
            except queue.Empty:
                app.update()
                continue
            started = time.perf_counter()
            context=None
            try:
                if time.monotonic() > command['expires']:
                    raise ValueError('Expired command; not applied')
                if command['payload']['sequence'] != sequence:
                    raise ValueError('Stale sequence; not applied')
                if command['path'] == '/step' and fault:
                    raise ValueError('Previous execution uncertain; reset required')
                if command['path'] == '/reset':
                    observation = control.reset(command['payload'].get('options'))
                    bank.set_baseline(observation)
                elif command['path']=='/step':
                    observation = control.step(command['payload'].get('action'))
                elif command['path']=='/save_reset':
                    context=bank.save(command['payload'].get('name'),command['payload'].get('provenance',{}))
                    observation=control.observe()
                else:
                    observation,context=bank.restore(command['payload'].get('name'))
                sequence += 1
                fault = False
                command['result'] = store.publish(observe_extra(observation), sequence,
                    {'physics_and_observe_ms': (time.perf_counter()-started)*1000},reset_context=context)
            except ValueError as error:
                command.update(result={'error': str(error)}, status=409)
            except Exception as error:
                fault = True
                sequence += 1
                command.update(result={'error': repr(error), 'sequence': sequence,
                                       'reset_required': True}, status=500)
                # Publish a new revision even when execution is uncertain.
                store.publish(control.observe(), sequence)
            finally:
                command['done'].set()
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if camera_capture:
            camera_capture.close()
        app.close()


if __name__ == '__main__':
    main()
