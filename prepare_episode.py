"""Export one local LeRobot episode for Isaac's NumPy-only replay reader."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path.home() / ".cache/huggingface/lerobot/Chojins/chess_game_001_blue_stereo")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/replay/episode_000000.npz"))
    args = parser.parse_args()
    info = json.loads((args.dataset / "meta/info.json").read_text())
    path = args.dataset / info["data_path"].format(episode_chunk=args.episode // info["chunks_size"], episode_index=args.episode)
    table = pq.read_table(path)
    state_names = [n.removeprefix("main_").removesuffix(".pos") for n in info["features"]["observation.state"]["names"]]
    action_names = [n.removeprefix("main_").removesuffix(".pos") for n in info["features"]["action"]["names"]]
    if state_names != action_names:
        raise ValueError("State and action joint order differ")
    states = np.array(table["observation.state"].to_pylist(), np.float64)
    actions = np.array(table["action"].to_pylist(), np.float64)
    timestamps = np.array(table["timestamp"].to_pylist(), np.float64).reshape(-1)
    if not all(np.isfinite(x).all() for x in (states, actions, timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Invalid episode values or timestamps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, states=states, actions=actions, timestamps=timestamps,
             joint_names=np.array(state_names), fps=info["fps"], source=str(path), episode_index=args.episode)
    print(f"Exported {len(states)} frames at {info['fps']} FPS to {args.output}")


if __name__ == "__main__":
    main()
