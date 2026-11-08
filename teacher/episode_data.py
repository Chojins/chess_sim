"""Lossless neural-teacher bundles and independent frame/action audits.

This format deliberately stays separate from the legacy residual/hybrid
collector: unified full-move scores are never renamed to pass its acceptance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from policy_overlay import highlight_move
from teacher.verify_task_space import audit_trial


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


class TeacherFrames:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);self.frames=0
        self.previous_sequence=None;self.previous_time=None
    def write(self,images,before,info,calibration,move):
        if not np.isfinite([before['simulation_time'],info['telemetry']['simulation_time']]).all():
            raise ValueError('Invalid physical time')
        if self.previous_sequence is not None and (before['sequence']!=self.previous_sequence
                or abs(before['simulation_time']-self.previous_time)>1e-6):
            raise ValueError('Lost frame between collection writes')
        if info['telemetry']['sequence']!=before['sequence']+1: raise ValueError('Lost camera/control sequence')
        if abs(info['telemetry']['simulation_time']-before['simulation_time']-1/30)>1e-6:
            raise ValueError('Lost physical timestep')
        state=np.asarray(before['observation']['observation_state'],np.float32)
        action=np.asarray(info['teacher']['executed_legacy'],np.float32)
        if state.shape!=(6,) or action.shape!=(6,) or not np.isfinite(np.r_[state,action]).all():
            raise ValueError('Invalid legacy state/action')
        for name in ('red','blue'):
            if images[name].dtype!=np.uint8 or images[name].shape!=(480,640,3): raise ValueError('Invalid RGB pair')
        red=np.ascontiguousarray(images['red'])
        laptop=np.ascontiguousarray(highlight_move(red[...,::-1].copy(),calibration,move)[...,::-1])
        np.savez_compressed(self.directory/f'frame_{self.frames:06d}.npz',
            laptop_rgb=laptop,phone_rgb=images['blue'],raw_red_rgb=red,
            observation_state=state,action_legacy=action,timestamp=np.float64(before['simulation_time']),
            frame_id=np.int64(before['sequence']),next_frame_id=np.int64(info['telemetry']['sequence']))
        self.frames+=1
        self.previous_sequence=info['telemetry']['sequence'];self.previous_time=info['telemetry']['simulation_time']


def audit_episode(directory):
    directory=Path(directory);record=json.loads((directory/'episode.json').read_text())
    if record.get('format')!='chess_sim_neural_teacher_episode_v1' or not record.get('accepted_for_act_training'):
        raise ValueError('Expected an accepted neural-teacher episode')
    manifest=json.loads((directory.parent/'manifest.json').read_text())
    if digest(manifest['bank'])!=manifest['bank_sha256']: raise ValueError('Collection reset bank changed')
    initial=json.loads((directory/'initial.json').read_text())
    if (manifest.get('teacher_label')!='neural_actor' or manifest.get('learned_policy') is not True
            or manifest.get('runtime_oracle') or manifest.get('oracle_interventions',0)
            or manifest.get('training_enabled') or initial['reset_context']['provenance']['family']!='home'
            or initial['reset_context']['provenance']['prior_lift_verified']
            or initial['reset_context']['provenance']['prior_release_verified']):
        raise ValueError('Collection requires a frozen neural actor from fresh home')
    if (record['provenance']['teacher_checkpoint_sha256']!=manifest['checkpoint_sha256']
            or digest(manifest['checkpoint'])!=manifest['checkpoint_sha256']):
        raise ValueError('Frozen teacher provenance changed')
    report=audit_trial(directory,manifest)
    from teacher.robot_clearance import append_clearance_audit
    report=append_clearance_audit(directory,manifest,report)
    if not report['success'] or not report['home_start']: raise ValueError('Physics did not verify a full home-start move')
    if report.get('robot_collision_audit',{}).get('contact_free') is False:
        raise ValueError('Robot collision prevents accepting this new-physics episode')
    if report.get('camera_clearance_audit',{}).get('passed') is False:
        raise ValueError('Insufficient camera/servo clearance prevents accepting this episode')
    rows=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
    frames=sorted((directory/'frames').glob('frame_*.npz'))
    if len(frames)!=len(rows) or record['frames']!=len(rows) or record['fps']!=30: raise ValueError('Incomplete lossless episode')
    previous=initial['telemetry'];hashes={}
    case=initial['reset_context']['provenance'].get('case',manifest['case'])
    calibration=manifest['camera_calibration'];move=case['move']
    for n,(path,row) in enumerate(zip(frames,rows)):
        if path.name!=f'frame_{n:06d}.npz': raise ValueError('Non-contiguous frame names')
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(f['observation_state'],np.asarray(previous['observation']['observation_state'],np.float32))
            np.testing.assert_array_equal(f['action_legacy'],np.asarray(row['teacher']['executed_legacy'],np.float32))
            if (int(f['frame_id'])!=previous['sequence'] or int(f['next_frame_id'])!=row['sequence']
                    or row['sequence']!=previous['sequence']+1): raise ValueError('Frame/action correspondence differs')
            if abs(float(f['timestamp'])-previous['simulation_time'])>1e-6: raise ValueError('Frame time differs')
            for key in ('laptop_rgb','phone_rgb','raw_red_rgb'):
                if f[key].shape!=(480,640,3) or f[key].dtype!=np.uint8: raise ValueError('Invalid lossless pixels')
            expected=highlight_move(f['raw_red_rgb'][...,::-1].copy(),calibration,move)[...,::-1]
            np.testing.assert_array_equal(f['laptop_rgb'],expected)
        hashes[str(path.relative_to(directory))]=digest(path)
        previous={'observation':row['observation'],'sequence':row['sequence'],'simulation_time':row['simulation_time']}
    if record['score']!=rows[-1]['score']: raise ValueError('Accepted score changed')
    # Verify every action really came from the frozen neural actor. This also
    # reconstructs actor inputs independently, with no HTTP/simulator mutation.
    from teacher.reconstruct import trial_transitions
    from teacher.grasp_checkpoints import ConservativeSAC
    model=ConservativeSAC.load(manifest['checkpoint'],device='cpu')
    for transition in trial_transitions(directory,manifest):
        predicted,_=model.predict(transition['observation'],deterministic=True)
        np.testing.assert_allclose(predicted,transition['action'],atol=1e-5,rtol=0)
    goal=json.loads((directory/'goal.json').read_text())
    if goal!=record['provenance']['goal'] or goal['uci']!=move: raise ValueError('Desired board provenance changed')
    import chess
    import chess.svg
    board=chess.Board(goal['initial_fen']);selected=chess.Move.from_uci(move)
    expected=chess.Board()
    for previous_move in case['history']: expected.push_uci(previous_move)
    if expected.fen()!=goal['initial_fen']: raise ValueError('Goal board differs from the requested history')
    if selected not in board.legal_moves: raise ValueError('Invalid goal move')
    board.push(selected)
    if board.fen()!=goal['desired_fen'] or chess.svg.board(board=board,lastmove=selected,orientation=chess.WHITE,size=480,coordinates=True)!=(directory/'desired_board.svg').read_text():
        raise ValueError('Desired board diagram differs from the legal move')
    for filename in ('initial.json','steps.jsonl','result.json','episode.json','goal.json','desired_board.svg'):
        hashes[filename]=digest(directory/filename)
    return {'path':str((directory/'frames').resolve()),'episode_path':str(directory.resolve()),'frames':len(rows),
        'move':move,'capture':False,'file_sha256':hashes,'provenance':record['provenance'],
        'collection_manifest_sha256':digest(directory.parent/'manifest.json')}
