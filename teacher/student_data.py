"""Episode-bounded RGB ACT chunks and local legacy demonstration replay."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np


def canonical_joint_names(names):
    # The v2.1 SO-100 dataset used a single-arm "main_" namespace.
    # Strip only that recorded namespace; never reorder or rescale values.
    return [name.removeprefix('main_') for name in names]


def chunk(actions, index, size):
    actions = np.asarray(actions, np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6 or not 0 <= index < len(actions) or size < 1:
        raise ValueError('Invalid episode chunk')
    positions = np.arange(index, index + size)
    return actions[np.minimum(positions, len(actions)-1)].copy(), positions >= len(actions)


def episode_split(episodes):
    """Hold out the last whole episode of each move, never adjacent frames."""
    groups = {}
    for episode in episodes:
        groups.setdefault(episode['move'], []).append(episode)
    if not groups or any(len(v) < 2 for v in groups.values()):
        raise ValueError('Need at least two complete episodes per move for an ACT pilot')
    training, validation = [], []
    for move in sorted(groups):
        values = sorted(groups[move], key=lambda e: e['episode_path'])
        training.extend(values[:-1]); validation.append(values[-1])
    return {'train': training, 'validation': validation}


def real_episode_split(numbers,training,validation,sampling='prefix'):
    numbers=list(numbers)
    if (min(training,validation)<1 or len(numbers)<training+validation or len(set(numbers))!=len(numbers)
            or sampling not in ('prefix','spread')):raise ValueError('Invalid whole-episode replay split')
    if sampling=='prefix':return {'train':numbers[:training],'validation':numbers[-validation:]}
    # Cover the recording sequence in both splits; no adjacent-frame split.
    held=[numbers[i] for i in np.linspace(0,len(numbers)-1,validation,dtype=int)]
    available=[n for n in numbers if n not in set(held)]
    selected=[available[i] for i in np.linspace(0,len(available)-1,training,dtype=int)]
    return {'train':selected,'validation':held}


class SimChunks:
    def __init__(self, index, chunk_size, audit=True):
        from teacher.episode_data import audit_episode, digest
        self.index = Path(index); self.digest = digest(index); self.chunk_size = chunk_size
        report = json.loads(self.index.read_text())
        if report.get('format') != 'chess_sim_neural_teacher_audited_v1':
            raise ValueError('Expected independently audited neural-teacher data')
        self.splits = episode_split(report['episodes']); self.cache = {}; self.goals = {}
        from teacher.goal_image import render_goal
        for episode in report['episodes']:
            if audit and audit_episode(episode['episode_path']) != episode:
                raise ValueError('Source episode changed since acceptance')
            directory = Path(episode['episode_path'])
            initial = json.loads((directory/'initial.json').read_text())['telemetry']['observation']
            rows = [json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
            states = [initial['observation_state']] + [r['observation']['observation_state'] for r in rows[:-1]]
            actions = [r['teacher']['executed_legacy'] for r in rows]
            self.cache[str(directory)] = (np.asarray(states, np.float32), np.asarray(actions, np.float32))
            self.goals[str(directory)] = render_goal(directory/'desired_board.svg')

    def sample(self, rng, split='train', index=None, episode=None):
        values = self.splits[split]
        if episode is None:
            moves = sorted({e['move'] for e in values}); move = moves[int(rng.integers(len(moves)))]
            choices = [e for e in values if e['move'] == move]; episode = choices[int(rng.integers(len(choices)))]
        if episode not in values: raise ValueError('Requested episode belongs to a different split')
        key = episode['episode_path']; states, actions = self.cache[key]
        if index is None: index = int(rng.integers(len(states)))
        with np.load(Path(episode['path'])/f'frame_{index:06d}.npz', allow_pickle=False) as f:
            np.testing.assert_array_equal(states[index], f['observation_state'])
            np.testing.assert_array_equal(actions[index], f['action_legacy'])
            images = (f['laptop_rgb'].copy(), f['phone_rgb'].copy(), self.goals[key])
        targets, padding = chunk(actions, index, self.chunk_size)
        return {'state': states[index], 'images': images, 'actions': targets, 'is_pad': padding,
                'episode': key, 'frame': index, 'move': episode['move']}


class RealChunks:
    """Read immutable local v2.1 data. No invented goal diagram for real replay."""
    def __init__(self, root, chunk_size, train_episodes=32, validation_episodes=4,sampling='prefix'):
        import pyarrow.parquet as pq
        self.root = Path(root); self.chunk_size = chunk_size; self.cache = {}; self.images = OrderedDict()
        self.info = json.loads((self.root/'meta/info.json').read_text())
        if self.info['codebase_version'] != 'v2.1' or self.info['fps'] != 30:
            raise ValueError('Legacy replay requires this explicit v2.1/30 Hz contract')
        if train_episodes < 1 or validation_episodes < 1: raise ValueError('Need disjoint real replay splits')
        numbers = [json.loads(s)['episode_index'] for s in (self.root/'meta/episodes.jsonl').read_text().splitlines()]
        if len(numbers) < train_episodes + validation_episodes: raise ValueError('Insufficient real episodes')
        self.splits=real_episode_split(numbers,train_episodes,validation_episodes,sampling)
        for number in self.splits['train'] + self.splits['validation']:
            path = self.path('data_path', number)
            table = pq.read_table(path, columns=['observation.state', 'action', 'timestamp', 'episode_index', 'frame_index'])
            if set(table['episode_index'].to_pylist()) != {number}: raise ValueError('Real episode identity differs')
            frames = table['frame_index'].to_numpy()
            np.testing.assert_array_equal(frames, np.arange(len(frames)))
            self.cache[number] = tuple(np.asarray(table[key].to_pylist(), np.float32)
                for key in ('observation.state', 'action', 'timestamp'))
            for camera in ('laptop', 'phone'):
                if self.path('video_path', number, camera).stat().st_size < 1024:
                    raise ValueError('Missing real video data')

    def path(self, kind, number, camera=None):
        return self.root/self.info[kind].format(episode_chunk=number//self.info['chunks_size'],
            episode_index=number, video_key='observation.images.'+str(camera))

    def image(self, number, camera, index, timestamp):
        import av
        key = (number, camera, index)
        if key in self.images:
            self.images.move_to_end(key); return self.images[key]
        with av.open(str(self.path('video_path', number, camera))) as container:
            stream = container.streams.video[0]; stream.thread_count = 1
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
            best = None
            for frame in container.decode(stream):
                time = float(frame.pts * stream.time_base); error = abs(time-timestamp)
                if best is None or error < best[0]: best = (error, frame)
                if time >= timestamp: break
            if best is None or best[0] > .5/30 + 1e-4:
                raise ValueError('Real RGB frame does not match the recorded joint timestamp')
            rgb = best[1].to_ndarray(format='rgb24')
        if rgb.shape != (480, 640, 3): raise ValueError('Wrong real RGB shape')
        self.images[key] = rgb
        while len(self.images) > 128: self.images.popitem(last=False)
        return rgb

    def sample(self, rng, split='train', index=None, episode=None):
        choices = self.splits[split]
        if episode is None: episode = choices[int(rng.integers(len(choices)))]
        if episode not in choices: raise ValueError('Real replay split leakage')
        states, actions, times = self.cache[episode]
        if index is None: index = int(rng.integers(len(states)))
        images = tuple(self.image(episode, c, index, float(times[index])) for c in ('laptop', 'phone'))
        targets, padding = chunk(actions, index, self.chunk_size)
        return {'state': states[index], 'images': images, 'actions': targets, 'is_pad': padding,
                'episode': episode, 'frame': index}
