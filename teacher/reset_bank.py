"""Simulator-local reset checkpoints; only reset may restore physical state.

Rigid poses/velocities and servo targets are restored. PhysX solver contact
caches are not serialized, so this is not a bitwise dynamics checkpoint.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ResetBank:
    version='chess_physics_reset_v1'

    def __init__(self,control,directory,layers):
        self.control=control
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        physics=copy.deepcopy(control.observe()['physics_settings'])
        if 'magnets' in physics:
            physics['magnets'].pop('initial_settling',None)
        self.identity={'format':self.version,'joint_map':control.mapping.config,
                       'piece_paths':list(control.driver.pieces.paths),'physics_settings':physics,
                       'layers':{str(Path(path).resolve()):digest(path) for path in sorted(set(layers)) if Path(path).is_file()},
                       'robot_dof_names':{color:list(robot.dof_names) for color,robot in control.driver.robots.items()}}
        manifest=self.directory/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=self.identity:
                raise ValueError('Reset bank scene, calibration or physics differs from the running simulator')
        else:
            manifest.write_text(json.dumps(self.identity,indent=2))
        self.baseline=None

    def set_baseline(self,observation):
        self.baseline=copy.deepcopy(observation)

    def path(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):
            raise ValueError('Invalid reset checkpoint name')
        return self.directory/(name+'.npz')

    def save(self,name,provenance):
        path=self.path(name)
        if path.exists():
            raise ValueError('Reset checkpoint already exists')
        if self.baseline is None:
            raise ValueError('A normal task reset is required before saving a checkpoint')
        if not isinstance(provenance,dict) or len(json.dumps(provenance))>2048:
            raise ValueError('Expected small reset provenance')
        control=self.control;driver=control.driver;pieces=driver.pieces
        observation=control.observe()
        if observation['task_geometry']!=self.baseline['task_geometry']:
            raise ValueError('Reset baseline and current task differ')
        positions,rotations=pieces.view.get_world_poses()
        arrays={'piece_positions':positions.copy(),'piece_rotations':rotations.copy(),
                'piece_velocities':pieces.view.get_velocities().copy(),
                'initial_piece_positions':pieces.initial_poses[0].copy(),
                'initial_piece_rotations':pieces.initial_poses[1].copy(),
                'initial_piece_centers':pieces.initial_centers.copy(),'maximum_lift':pieces.maximum_lift.copy(),
                'last_target':control.last_target.copy()}
        for color,robot in driver.robots.items():
            applied=robot.get_applied_actions()
            arrays[color+'_q']=robot.get_joint_positions().copy()
            arrays[color+'_qd']=robot.get_joint_velocities().copy()
            arrays[color+'_targets']=np.asarray(applied.joint_positions).copy()
            arrays[color+'_velocity_targets']=np.asarray(applied.joint_velocities).copy()
        metadata={'format':self.version,'name':name,'identity':self.identity,
                  'baseline_observation':self.baseline,'saved_observation':observation,
                  'task_geometry':control.task_geometry,'reset_options':control.reset_options,
                  'active_index':pieces.active_index,'provenance':provenance,
                  'limitations':'PhysX contact solver warm-start caches are not serialized'}
        self.validate(arrays)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata_json=np.array(json.dumps(metadata)),**arrays)
        return {'name':name,'path':str(path.resolve()),'sha256':digest(path),
                'saved_control_step':observation['control_step']}

    def validate(self,arrays):
        count=len(self.control.driver.pieces.paths)
        shapes={'piece_positions':(count,3),'piece_rotations':(count,4),'piece_velocities':(count,6),
                'initial_piece_positions':(count,3),'initial_piece_rotations':(count,4),
                'initial_piece_centers':(count,3),'maximum_lift':(count,),'last_target':(6,)}
        for color,robot in self.control.driver.robots.items():
            shape=robot.get_joint_positions().shape
            for field in ('q','qd','targets','velocity_targets'):
                shapes[color+'_'+field]=shape
        if set(arrays)!=set(shapes):
            raise ValueError('Incomplete reset checkpoint')
        for name,shape in shapes.items():
            if arrays[name].shape!=shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f'Invalid reset array: {name}')
        for name in ('piece_rotations','initial_piece_rotations'):
            if not np.allclose(np.linalg.norm(arrays[name],axis=1),1.,atol=1e-4):
                raise ValueError('Invalid piece quaternion')

    def restore(self,name):
        path=self.path(name)
        if not path.is_file():
            raise ValueError('Unknown reset checkpoint')
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata_json']))
            arrays={key:archive[key].copy() for key in archive.files if key!='metadata_json'}
        if metadata['format']!=self.version or metadata['identity']!=self.identity or metadata['name']!=name:
            raise ValueError('Checkpoint identity mismatch')
        self.validate(arrays)  # Validate every array before any mutation.
        control=self.control;driver=control.driver;pieces=driver.pieces
        driver.world.pause()
        for color,robot in driver.robots.items():
            robot.set_joint_positions(arrays[color+'_q'])
            robot.set_joint_velocities(arrays[color+'_qd'])
            robot.set_joint_position_targets(arrays[color+'_targets'])
            robot.set_joint_velocity_targets(arrays[color+'_velocity_targets'])
        pieces.view.set_world_poses(arrays['piece_positions'],arrays['piece_rotations'])
        pieces.view.set_velocities(arrays['piece_velocities'])
        pieces.initial_poses=(arrays['initial_piece_positions'],arrays['initial_piece_rotations'])
        pieces.initial_centers=arrays['initial_piece_centers']
        pieces.maximum_lift=arrays['maximum_lift']
        pieces.active_index=metadata['active_index']
        control.task_geometry=copy.deepcopy(metadata['task_geometry'])
        control.reset_options=copy.deepcopy(metadata['reset_options'])
        control.last_target=arrays['last_target']
        control.episode_id+=1;control.step_id=0
        control.start_time=driver.world.current_time
        driver.tracking.clear()
        self.baseline=copy.deepcopy(metadata['baseline_observation'])
        observed=control.observe()
        current_positions,current_rotations=pieces.view.get_world_poses()
        parity={'max_piece_position_error_m':float(np.max(np.abs(current_positions-arrays['piece_positions']))),
                'max_piece_rotation_error':float(np.max(np.abs(current_rotations-arrays['piece_rotations']))),
                'max_piece_velocity_error':float(np.max(np.abs(pieces.view.get_velocities()-arrays['piece_velocities']))),
                'max_robot_joint_error_rad':max(float(np.max(np.abs(robot.get_joint_positions()-arrays[color+'_q'])))
                                               for color,robot in driver.robots.items()),
                'max_robot_velocity_error_rad_s':max(float(np.max(np.abs(robot.get_joint_velocities()-arrays[color+'_qd'])))
                                               for color,robot in driver.robots.items())}
        if max(parity.values())>1e-5:
            raise RuntimeError('Physical reset did not restore the recorded state within tolerance')
        context={'name':name,'checkpoint_sha256':digest(path),'baseline_observation':self.baseline,
                 'provenance':metadata['provenance'],'saved_control_step':metadata['saved_observation']['control_step'],
                 'parity':parity,'limitations':metadata['limitations']}
        return observed,context
