"""Goal-conditioned, privileged-state teacher with no ACT or capture macro.

The actor emits six normalized absolute physical joint targets. A generic slew
limit bounds target velocity; task phases never replace the actor's commands.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_task import LegalMoveTask, TaskConfig
from review_grasp_geometry import opening_center
from teacher.transport import StateClient

ASSETS = Path(__file__).parent / 'assets'
CASE = {'id': 'scandinavian_capture', 'history': ['e2e4', 'd7d5'], 'move': 'e4d5'}


@dataclass(frozen=True)
class TeacherConfig:
    max_steps: int = 1200
    gamma: float = .99
    jitter_mm: float = .5
    max_velocity_deg_s: tuple = (60., 60., 60., 60., 60., 60.)
    shaping_weight: float = .5
    acceleration_weight: float = .005
    velocity_weight: float = .002
    action_mode: str = 'absolute'
    acceleration_cost_cap: float | None = None

    def __post_init__(self):
        values = [self.jitter_mm, self.shaping_weight, self.acceleration_weight, self.velocity_weight]
        if (type(self.max_steps) is not int or self.max_steps < 1 or not 0 < self.gamma < 1
                or not np.isfinite(values).all() or min(values) < 0 or self.jitter_mm > 2
                or np.asarray(self.max_velocity_deg_s).shape != (6,)
                or not np.isfinite(self.max_velocity_deg_s).all() or min(self.max_velocity_deg_s) <= 0
                or self.action_mode not in ('absolute','delta','arm_delta_gripper_position')
                or self.acceleration_cost_cap is not None and (
                    not np.isfinite(self.acceleration_cost_cap) or self.acceleration_cost_cap <= 0)):
            raise ValueError('Invalid teacher configuration')


class TeacherFeatures:
    version = 'privileged_chess_goal_v1'

    def __init__(self, initial, specification, anchors, config):
        self.spec, self.anchors, self.config = specification, anchors, config
        self.geometry = initial['task_geometry']
        self.home = np.asarray(initial['measured_joint_deg'], float)
        self.paths = list(initial['pieces']['piece_paths'])
        if len(self.paths) != 32 or len(set(self.paths)) != 32:
            raise ValueError('Expected the fixed 32-piece inventory')
        self.goals = np.array(initial['pieces']['piece_centers_m'], float)
        self.roles = np.zeros((32, 2))
        source, victim = self.geometry['source_index'], self.geometry['capture_index']
        self.roles[source, 0] = 1
        self.goals[source] = self.geometry['target_center_m']
        if victim is not None:
            self.roles[victim, 1] = 1
            zone = self.geometry['capture_zone']
            lo, hi = np.array(zone['min_xy_m']), np.array(zone['max_xy_m'])
            self.goals[victim] = [lo[0]+min(.035, (hi[0]-lo[0])/2),
                                  (lo[1]+hi[1])/2, self.geometry['capture_rest_center_z_m']]
        self.origin = np.mean(self.geometry['square_centers_m'], axis=0)
        self.schema = {'joints_deg_div180': 6, 'joint_velocity_deg_s_div180': 6,
                       'command_deg_div180': 6, 'command_velocity_scaled': 6,
                       'home_deg_div180': 6, 'opening_world_offset_scaled': 3,
                       'finger_angles_deg_div45': 2, 'pieces_center_relative_and_upright': 32*7,
                       'desired_minus_actual_piece_centers_scaled': 32*3,
                       'piece_roles_source_victim': 32*2, 'progress_and_time': 5}
        self.size = sum(self.schema.values())

    def point(self, observation):
        return opening_center(self.spec, self.anchors, observation['measured_joint_deg'],
                              observation['measured_finger_joint_deg']['jaw2_hinge'])

    def encode(self, observation, task, command, velocity, previous_q):
        if observation['pieces']['piece_paths'] != self.paths:
            raise ValueError('Piece inventory/order changed')
        q = np.asarray(observation['measured_joint_deg'])
        centers = np.asarray(observation['pieces']['piece_centers_m'])
        point = self.point(observation)
        upright = np.asarray(observation['pieces']['piece_upright_cosines'])[:, None]
        pieces = np.c_[(centers-self.origin)/.25, (centers-point)/.25, upright]
        fingers = observation['measured_finger_joint_deg']
        values = np.r_[q/180, (q-previous_q)*30/180, np.asarray(command)/180,
                       np.asarray(velocity)/self.config.max_velocity_deg_s, self.home/180,
                       (point-self.origin)/.25,
                       [fingers['jaw1_hinge']/45, fingers['jaw2_hinge']/45], pieces.ravel(),
                       ((self.goals-centers)/.25).ravel(), self.roles.ravel(),
                       [float(task.capture_index is not None), float(task.capture_bonus),
                        float(task.capture_lifted), float(task.lifted), task.step_count/self.config.max_steps]]
        result = np.asarray(values, np.float32)
        if result.shape != (self.size,) or not np.isfinite(result).all():
            raise ValueError('Invalid privileged teacher features')
        return result

    def potential(self, observation, task):
        centers = np.asarray(observation['pieces']['piece_centers_m'])
        target_index = task.capture_index if task.capture_index is not None and not task.capture_bonus else task.source_index
        grasp = np.asarray(observation['pieces']['piece_grasp_targets_m'][target_index])
        approach = np.exp(-np.sum(((self.point(observation)-grasp)/.04)**2))
        lifted = task.capture_lifted if target_index == task.capture_index else task.lifted
        placement = float(lifted)*np.exp(-np.sum(((centers[target_index]-self.goals[target_index])/.05)**2))
        return float(4*task.capture_bonus+approach+2*placement)


class StateTeacherEnv(gym.Env):
    def __init__(self, url='http://127.0.0.1:8081', config=None, case=None, client=None,
                 kinematics=ASSETS/'kinematics.json', reference=ASSETS/'finger_reference.json'):
        self.config = config or TeacherConfig()
        self.case = dict(CASE if case is None else case)
        self.client = client or StateClient(url)
        self.kinematics = json.loads(Path(kinematics).read_text())
        reference_data = json.loads(Path(reference).read_text())
        self.grasp_reference = reference_data['grasp_reference']
        if self.kinematics['usd_sha256'] != reference_data['usd_sha256']:
            raise ValueError('Kinematics and approved finger geometry disagree')
        import hashlib
        if hashlib.sha256(Path('Sim_Stage.usd').read_bytes()).hexdigest() != self.kinematics['usd_sha256']:
            raise ValueError('Authored scene changed since kinematics export')
        self.anchors = reference_data['anchors_jaw_local_m']
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self.needs_reset = True
        obs, _ = self.reset(seed=0)
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Use the explicit case/config to choose the task')
        self.needs_reset = True
        self.latest = self.client.reset({'history': self.case['history'], 'move': self.case['move'],
                                         'pawn_offset_mm': self.np_random.uniform(-self.config.jitter_mm, self.config.jitter_mm, 2).tolist()})
        self.actual = self.latest['observation']
        if self.actual.get('grasp_reference') != self.grasp_reference:
            raise ValueError('Server grasp geometry differs from the approved teacher reference')
        if not np.allclose(self.actual['joint_limits_deg'], self.kinematics['limits'], atol=1e-5):
            raise ValueError('Server joint limits differ from exported kinematics')
        self.task = LegalMoveTask(self.actual, TaskConfig(max_steps=self.config.max_steps, capture_home_retry=True))
        self.encoder = TeacherFeatures(self.actual, self.kinematics, self.anchors, self.config)
        self.limits = np.asarray(self.actual['joint_limits_deg'], float)
        self.low, self.high = self.limits[:, 0]+1e-4, self.limits[:, 1]-1e-4
        self.mapping = self.actual['joint_map']
        self.command = np.clip(np.asarray(self.actual['target_legacy_action'])*self.mapping['scale']
                               +np.asarray(self.mapping['sim_home_deg'])
                               -np.asarray(self.mapping['legacy_home'])*self.mapping['scale'], self.low, self.high)
        self.previous_q = np.asarray(self.actual['measured_joint_deg'])
        self.velocity = np.zeros(6)
        self.previous_potential = self.encoder.potential(self.actual, self.task)
        self.needs_reset = False
        return self.encoder.encode(self.actual, self.task, self.command, self.velocity, self.previous_q), {
            'telemetry': self.latest, 'task_version': self.task.version}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('reset() is required before step()')
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action) or not np.isfinite(action).all():
            raise ValueError('Expected six finite normalized joint targets')
        max_step = np.asarray(self.config.max_velocity_deg_s)/30
        desired = (self.command+action.astype(float)*max_step if self.config.action_mode!='absolute' else
                   self.low+(action.astype(float)+1)*.5*(self.high-self.low))
        if self.config.action_mode=='arm_delta_gripper_position':
            desired[5]=self.low[5]+(float(action[5])+1)*.5*(self.high[5]-self.low[5])
        command = np.clip(self.command+np.clip(desired-self.command, -max_step, max_step), self.low, self.high)
        velocity = (command-self.command)*30
        legacy = np.asarray(self.mapping['legacy_home'])+(command-self.mapping['sim_home_deg'])/self.mapping['scale']
        self.needs_reset = True  # An uncertain command never becomes a fabricated transition.
        self.latest = self.client.step(legacy)
        previous_q = np.asarray(self.actual['measured_joint_deg'])
        self.actual = self.latest['observation']
        reward, terminated, truncated, score = self.task.update(self.actual)
        potential = self.encoder.potential(self.actual, self.task)
        # True terminal states have zero potential. Time limits still bootstrap.
        shaping = self.config.shaping_weight*(self.config.gamma*(0 if terminated else potential)-self.previous_potential)
        acceleration = (velocity-self.velocity)*30
        penalties = dict(command_acceleration=self.config.acceleration_weight*float(np.mean((acceleration/180)**2)),
                         command_velocity=self.config.velocity_weight*float(np.mean((velocity/60)**2)))
        if self.config.acceleration_cost_cap is not None:
            penalties['command_acceleration'] = min(penalties['command_acceleration'],self.config.acceleration_cost_cap)
        result = self.encoder.encode(self.actual, self.task, command, velocity, previous_q)
        self.command, self.velocity, self.previous_q = command, velocity, previous_q
        self.previous_potential = potential
        self.needs_reset = terminated or truncated
        return result, float(reward+shaping-sum(penalties.values())), terminated, truncated, {
            'telemetry': self.latest, 'score': score, 'is_success': score['success'],
            'teacher': {'executed_legacy': legacy.tolist(), 'command_joint_deg': command.tolist(),
                        'command_velocity_deg_s': velocity.tolist(), 'command_acceleration_deg_s2': acceleration.tolist(),
                        'potential': potential, 'shaping': shaping, 'penalties': penalties,
                        'privileged_actor': True, 'act_or_capture_supervisor': False}}

    def close(self):
        self.needs_reset = True
        self.client.close()

    def normalized_target(self, target):
        """Invert the selected action contract for an explicit physical target."""
        target=np.clip(np.asarray(target,float),self.low,self.high)
        action=((target-self.command)*30/self.config.max_velocity_deg_s if self.config.action_mode!='absolute'
                else 2*(target-self.low)/(self.high-self.low)-1)
        if self.config.action_mode=='arm_delta_gripper_position':
            action[5]=2*(target[5]-self.low[5])/(self.high[5]-self.low[5])-1
        return np.clip(action,-1,1).astype(np.float32)

    def manifest(self):
        return {'observation_version': self.encoder.version, 'observation_schema': self.encoder.schema,
                'observation_size': self.encoder.size, 'config': asdict(self.config), 'case': self.case,
                'joint_map': self.mapping, 'joint_limits_deg': self.limits.tolist(),
                'action': ('arm joint increments and normalized gripper position; all outputs learned'
                           if self.config.action_mode=='arm_delta_gripper_position' else
                           'normalized physical joint target increment, zero holds the previous command'
                           if self.config.action_mode=='delta' else
                           'normalized absolute physical joint target, generic 30 Hz target-velocity limit'),
                'actor_inputs': 'privileged geometry, joint feedback, command history and goal; no images',
                'grasp_reference': self.grasp_reference,
                'act_or_capture_supervisor': False, 'physics_settings': self.actual['physics_settings']}
