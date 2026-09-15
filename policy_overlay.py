"""The 2-D chess_vision overlay without its hardware/3-D import dependencies."""
import re
import cv2
import numpy as np


def highlight_move(bgr, calibration, move):
    if not re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', move):
        raise ValueError('Expected a UCI chess move')
    if list(bgr.shape[:2][::-1]) != calibration['resolution']:
        raise ValueError('Camera resolution differs from board calibration')
    size, square = calibration['board_size'], calibration['square_size']
    corners = np.float32(calibration['inner_corners'])
    dest = np.float32([[square, square], [7 * square, square], [7 * square, 7 * square], [square, 7 * square]])
    matrix = cv2.getPerspectiveTransform(corners, dest)
    inverse = cv2.getPerspectiveTransform(dest, corners)
    warped = cv2.warpPerspective(bgr, matrix, (size, size))
    overlay = warped.copy()
    for text, color in [(move[:2], (0, 255, 255)), (move[2:4], (0, 255, 0))]:
        x, y = (ord(text[0]) - ord('a')) * square, (8 - int(text[1])) * square
        cv2.rectangle(overlay, (x, y), (x + square, y + square), color, -1)
    blended = cv2.addWeighted(overlay, 0.3, warped, 0.7, 0)
    result = cv2.warpPerspective(blended, inverse, (bgr.shape[1], bgr.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(warped), inverse, (bgr.shape[1], bgr.shape[0]))
    return np.where(mask > 0, result, bgr)
