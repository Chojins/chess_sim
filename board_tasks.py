"""Legal, history-based chess scene plans. No USD or robot dependencies."""
import chess
import numpy as np


class BoardLayout:
    def __init__(self, paths, centers, squares, board_top, board_bounds=None):
        self.paths = list(paths)
        self.centers = np.asarray(centers).copy()
        sites = sorted(np.asarray(squares), key=lambda p: p[1])
        if len(sites) != 64:
            raise ValueError('Expected 64 square centres')
        self.squares = np.array([p for start in range(0, 64, 8)
                                for p in sorted(sites[start:start + 8], key=lambda p: p[0])])
        self.heights = self.centers[:, 2] - board_top
        self.start = {}
        board = chess.Board()
        for color, group in [(chess.WHITE, 'blue'), (chess.BLACK, 'red')]:
            for kind in range(1, 7):
                candidates = [i for i, path in enumerate(paths)
                              if f'chess_pieces_{group}/' in path and path.split('/')[-1].split('_')[0] == chess.piece_name(kind)]
                targets = sorted(board.pieces(kind, color))
                if len(candidates) != len(targets):
                    raise ValueError(f'Asset inventory mismatch: {group} {chess.piece_name(kind)}')
                for square in targets:
                    index = min(candidates, key=lambda i: np.linalg.norm(self.centers[i, :2] - self.squares[square, :2]))
                    self.start[square] = index
                    candidates.remove(index)
        lo, hi = self.squares[:, :2].min(0), self.squares[:, :2].max(0)
        if board_bounds is not None:
            lo, hi = np.asarray(board_bounds)[:, :2]
        else:
            lo, hi = lo - .01125, hi + .01125
        middle_y = float((lo[1] + hi[1]) / 2)
        self.zones = {
            'blue': {'min_xy_m': [float(hi[0]), middle_y - .075],
                     'max_xy_m': [float(hi[0] + .120), middle_y + .075], 'surface_z_m': 0.},
            'red': {'min_xy_m': [float(lo[0] - .120), middle_y - .075],
                    'max_xy_m': [float(lo[0]), middle_y + .075], 'surface_z_m': 0.}}
        for color in self.zones:
            self.zones[color].update(robot=color, calibrated=False, convention='acting robot right, facing board')

    def plan(self, history, move_text):
        if not isinstance(history, list) or len(history) > 300 or not all(isinstance(m, str) for m in history):
            raise ValueError('history must be a list of at most 300 legal UCI moves')
        board, occupied = chess.Board(), self.start.copy()
        for text in history:
            try:
                move = board.parse_uci(text)
            except (ValueError, TypeError) as error:
                raise ValueError(f'Illegal history move {text}') from error
            if move.promotion:
                raise ValueError('Promotion history requires additional piece assets; not yet supported')
            if board.is_en_passant(move):
                occupied.pop(move.to_square - 8 if board.turn else move.to_square + 8)
            if board.is_castling(move):
                rank = chess.square_rank(move.from_square)
                rook_from, rook_to = ((7, 5) if chess.square_file(move.to_square) > 4 else (0, 3))
                occupied[chess.square(rook_to, rank)] = occupied.pop(chess.square(rook_from, rank))
            occupied[move.to_square] = occupied.pop(move.from_square)
            board.push(move)
        try:
            move = board.parse_uci(move_text)
        except (ValueError, TypeError) as error:
            raise ValueError('Requested move is not legal in this position') from error
        if move.promotion or board.is_castling(move):
            raise ValueError('Castling/promotion execution needs multi-piece scoring; not yet supported')
        actor = 'blue' if board.turn else 'red'
        source_index = occupied[move.from_square]
        capture_square = (move.to_square - 8 if board.turn else move.to_square + 8) if board.is_en_passant(move) else move.to_square
        captured_index = occupied.get(capture_square) if board.is_capture(move) else None
        centers = self.centers.copy()
        for square, index in occupied.items():
            centers[index, :2] = self.squares[square, :2]
        parked_counts = {'blue': 0, 'red': 0}
        for index, path in enumerate(self.paths):
            if index in occupied.values():
                continue
            captor = 'red' if 'chess_pieces_blue/' in path else 'blue'
            slot = parked_counts[captor]
            parked_counts[captor] += 1
            zone = self.zones[captor]
            lo, hi = np.array(zone['min_xy_m']), np.array(zone['max_xy_m'])
            centers[index, :2] = lo + (hi - lo) * [(slot % 3 + .5) / 3, (slot // 3 + .5) / 5]
            centers[index, 2] = zone['surface_z_m'] + self.heights[index]
        source = centers[source_index].copy()
        target = source.copy()
        target[:2] = self.squares[move.to_square, :2]
        after = board.copy()
        after.push(move)
        geometry = {'version': 'legal_move_v1', 'move': move.uci(), 'history': history,
                    'square_centers_m': self.squares.tolist(),
                    'fen_before': board.fen(), 'fen_after_expected': after.fen(), 'acting_robot': actor,
                    'source_path': self.paths[source_index], 'source_index': source_index,
                    'source_center_m': source.tolist(), 'target_center_m': target.tolist(),
                    'capture_index': captured_index, 'capture_path': None if captured_index is None else self.paths[captured_index],
                    'capture_zone': self.zones[actor], 'capture_zones': self.zones,
                    'capture_rest_center_z_m': None if captured_index is None else self.zones[actor]['surface_z_m'] + float(self.heights[captured_index]),
                    'en_passant': board.is_en_passant(move)}
        return centers, geometry
