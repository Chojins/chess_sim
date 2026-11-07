"""Save an explicit desired board for later goal-image conditioning."""
import json
from pathlib import Path
import chess
import chess.svg


def save_goal(directory,initial_fen,uci):
    board=chess.Board(initial_fen);move=chess.Move.from_uci(uci)
    if move not in board.legal_moves: raise ValueError('Desired move is illegal in its initial board')
    board.push(move)
    svg=chess.svg.board(board=board,lastmove=move,orientation=chess.WHITE,size=480,coordinates=True)
    directory=Path(directory)
    with (directory/'desired_board.svg').open('x') as stream: stream.write(svg)
    goal={'initial_fen':initial_fen,'desired_fen':board.fen(),'uci':uci,
        'diagram':'desired_board.svg','diagram_orientation':'white at bottom',
        'student_input_status':'Recorded for future goal-image conditioning; current ACT export retains its two original RGB inputs'}
    with (directory/'goal.json').open('x') as stream: json.dump(goal,stream,indent=2)
    return goal
