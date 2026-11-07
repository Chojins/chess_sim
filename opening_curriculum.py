"""Build replayable legal opening tasks using the existing Stockfish workflow."""
import argparse
import json
from pathlib import Path

import chess
import chess.engine


def task(history, move, name):
    board = chess.Board()
    for text in history:
        board.push(board.parse_uci(text))
    action = board.parse_uci(move)
    return {'id': name, 'history': history.copy(), 'move': action.uci(), 'fen': board.fen(),
            'acting_robot': 'blue' if board.turn else 'red', 'capture': board.is_capture(action),
            'castling': board.is_castling(action), 'promotion': action.promotion,
            'executable': bool(board.turn and not board.is_castling(action) and not action.promotion)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', default='/usr/games/stockfish')
    parser.add_argument('--plies', type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.plies <= 40:
        parser.error('plies must be between 2 and 40')
    # Named probes remain stable even when engine search varies by version.
    tasks = [task([], 'e2e4', 'initial_e2e4'), task([], 'd2d4', 'initial_d2d4'),
             task(['e2e4', 'e7e5'], 'g1f3', 'open_game_knight'),
             task(['e2e4', 'd7d5'], 'e4d5', 'scandinavian_capture')]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        for first in ('e2e4', 'd2d4', 'c2c4'):
            board, history = chess.Board(), []
            for ply in range(args.plies):
                if board.is_game_over(): break
                move = board.parse_uci(first) if not history else engine.play(board, chess.engine.Limit(time=.1)).move
                tasks.append(task(history, move.uci(), f'{first}_ply_{ply + 1:02}'))
                board.push(move)
                history.append(move.uci())
        report = {'engine': engine.id, 'engine_path': args.engine, 'search_time_s': .1,
                  'scope': 'Chosen opening seeds plus Stockfish self-play, not claimed to match human opening frequencies',
                  'tasks': tasks}
    finally:
        engine.quit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(f'Saved {len(tasks)} legal tasks; {sum(t["executable"] for t in tasks)} executable with current blue calibration')


if __name__ == '__main__':
    main()
