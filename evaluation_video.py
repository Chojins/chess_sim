"""Record synchronized fixed/wrist camera frames as browser-compatible H.264."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


class EpisodeVideo:
    def __init__(self, directory, images, fps=30):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise RuntimeError('Video recording requires ffmpeg and ffprobe')
        self.path = Path(directory) / 'cameras.mp4'
        frame = self.frame(images)
        self.shape = frame.shape
        self.frames = 0
        self.log = (Path(directory) / 'video_encoder.log').open('wb')
        h, w = frame.shape[:2]
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.path)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log)

    @staticmethod
    def frame(images):
        left, right = images['laptop_rgb'], images['phone_rgb']
        if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError('Video cameras must have matching uint8 RGB frames')
        return np.ascontiguousarray(np.concatenate([left, right], axis=1))

    def write(self, images):
        frame = self.frame(images)
        if frame.shape != self.shape:
            raise ValueError('Video frame size changed')
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError('Video encoder did not finish')
        finally:
            self.log.close()
        if code:
            raise RuntimeError(f'Video encoding failed; see {self.log.name}')

    def verify(self):
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
                                '-select_streams', 'v:0', '-show_entries',
                                'stream=codec_name,width,height,nb_read_frames',
                                '-of', 'json', str(self.path)], check=True,
                               capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        assert stream['codec_name'] == 'h264'
        assert int(stream['nb_read_frames']) == self.frames
        assert (stream['height'], stream['width']) == self.shape[:2]
        return {'path': self.path.name, 'frames': self.frames,
                'fps': 30, 'left': 'fixed camera with move overlay',
                'right': 'moving wrist camera', 'codec': 'h264',
                'playback': 'simulation time; frame 0 is reset, frame N is step N'}


def write_training_video_index(root):
    import html
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else {}
    objective = manifest.get('objective', 'See saved manifest for scoring objective')
    body = ['<!doctype html><meta charset="utf-8"><title>Policy recordings</title>',
            '<h1>Policy recordings</h1>',
            f'<p>Objective: {html.escape(objective)}. Left: fixed camera; right: wrist camera. Videos play at simulated 30 Hz. Refresh for completed episodes.</p><ul>']
    for result_path in sorted(root.glob('*/*/result.json')):
        row = json.loads(result_path.read_text())
        if 'video' not in row:
            continue
        relative = result_path.parent.relative_to(root) / row['video']['path']
        title = f"{result_path.parent.parent.name}: {row['label']}, seed {row['seed']} — {row['score']['reason']}"
        if 'collection' in row:
            title += ' — ' + ('accepted for ACT data' if row['collection']['accepted_for_act_training'] else 'excluded from ACT data')
        body.append(f'<li><a href="{html.escape(str(relative))}">{html.escape(title)}</a></li>')
    body.append('</ul>')
    (root / 'videos.html').write_text('\n'.join(body))
