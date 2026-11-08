"""Goal-conditioned blue pawn/knight moves with the original completion rules.

The original e2e4 task stays strict and retains its version. This task explicitly
validates legal board history/move and checks each restored goal against its
requested case. Captures, promotions, castling and red actions are separate work.
"""
import hashlib
import chess
from teacher.move_teacher import MoveTask
from teacher.task_space import TaskSpaceMoveEnv


def ordinary_case(history,move):
    if not isinstance(history,list) or not all(isinstance(m,str) for m in history):
        raise ValueError('Need an explicit list of legal UCI history moves')
    board=chess.Board()
    try:
        for text in history:
            played=chess.Move.from_uci(text)
            if played not in board.legal_moves: raise ValueError('Illegal board history')
            board.push(played)
        selected=chess.Move.from_uci(move)
        if selected not in board.legal_moves: raise ValueError('Requested move is illegal')
    except (ValueError,TypeError) as error:
        raise ValueError('Invalid ordinary move/history') from error
    piece=board.piece_at(selected.from_square)
    if (board.turn!=chess.WHITE or piece is None or piece.piece_type not in (chess.PAWN,chess.KNIGHT)
            or board.is_capture(selected) or selected.promotion or board.is_castling(selected)):
        raise ValueError('Only non-capturing blue pawn/knight moves are supported')
    prefix='initial' if not history else hashlib.sha256(board.fen().encode()).hexdigest()[:12]
    return {'id':f'{prefix}_{move}','history':list(history),'move':move}


def validate_case_geometry(case,geometry):
    expected=ordinary_case(case['history'],case['move'])
    actual=ordinary_case(geometry.get('history',[]),geometry['move'])
    if (expected!=actual or geometry.get('acting_robot')!='blue' or geometry['capture_index'] is not None):
        raise ValueError('Physical goal differs from the requested ordinary move')
    return expected


class OrdinaryMoveTask(MoveTask):
    version='ordinary_move_v1'
    objective='ordinary_move'
    tipped_reason='piece_tipped'
    def validate_geometry(self,geometry):
        validate_case_geometry(ordinary_case(geometry.get('history',[]),geometry['move']),geometry)


class OrdinaryTaskSpaceEnv(TaskSpaceMoveEnv):
    task_type=OrdinaryMoveTask
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for entry in self.bank['entries']:
            ordinary_case(entry['case']['history'],entry['case']['move'])
    def reset(self,*args,**kwargs):
        obs,initial=super().reset(*args,**kwargs)
        validate_case_geometry(self.entry['case'],self.base.actual['task_geometry'])
        if self.entry['case']!=self.base.latest['reset_context']['provenance']['case']:
            raise ValueError('Checkpoint case provenance changed')
        return obs,initial
