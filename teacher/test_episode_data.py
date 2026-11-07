import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import chess
from teacher.episode_data import TeacherFrames
from teacher.collect_episodes import qualification
from teacher.goal_spec import save_goal


class TeacherDataTests(unittest.TestCase):
    def inputs(self):
        images={'red':np.full((480,640,3),[35,70,110],np.uint8),'blue':np.full((480,640,3),175,np.uint8)}
        before={'sequence':8,'simulation_time':0.,'observation':{'observation_state':[1.,2,3,4,5,6]}}
        info={'telemetry':{'sequence':9,'simulation_time':1/30},'teacher':{'executed_legacy':[2.,3,4,5,6,7]}}
        calibration={'resolution':[640,480],'board_size':400,'square_size':50,
            'inner_corners':[[100,100],[400,100],[400,400],[100,400]]}
        return images,before,info,calibration
    def test_lossless_before_pair_and_executed_legacy_targets(self):
        with TemporaryDirectory() as tmp:
            writer=TeacherFrames(Path(tmp)/'frames');images,before,info,calibration=self.inputs()
            writer.write(images,before,info,calibration,'e2e4')
            with np.load(writer.directory/'frame_000000.npz',allow_pickle=False) as frame:
                np.testing.assert_array_equal(frame['phone_rgb'],images['blue'])
                np.testing.assert_array_equal(frame['raw_red_rgb'],images['red'])
                np.testing.assert_array_equal(frame['observation_state'],before['observation']['observation_state'])
                np.testing.assert_array_equal(frame['action_legacy'],info['teacher']['executed_legacy'])
                self.assertEqual(int(frame['frame_id']),8);self.assertEqual(int(frame['next_frame_id']),9)
                self.assertTrue(np.any(frame['laptop_rgb']!=images['red']))
    def test_skipped_sequence_or_time_rejected_without_frame(self):
        with TemporaryDirectory() as tmp:
            writer=TeacherFrames(Path(tmp)/'frames');images,before,info,calibration=self.inputs()
            info['telemetry']['sequence']=10
            with self.assertRaises(ValueError): writer.write(images,before,info,calibration,'e2e4')
            self.assertEqual(writer.frames,0)
    def test_loss_between_writes_and_nonfinite_time_rejected(self):
        with TemporaryDirectory() as tmp:
            writer=TeacherFrames(Path(tmp)/'frames');images,before,info,calibration=self.inputs()
            writer.write(images,before,info,calibration,'e2e4')
            before['sequence']=10;before['simulation_time']=2/30;info['telemetry']={'sequence':11,'simulation_time':3/30}
            with self.assertRaises(ValueError): writer.write(images,before,info,calibration,'e2e4')
            before['simulation_time']=float('nan')
            with self.assertRaises(ValueError): writer.write(images,before,info,calibration,'e2e4')
    def test_partial_or_failed_qualification_cannot_collect(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text('{}')
            for report in ({'passed':False},{'passed':True,'stage':'rl_training'}):
                (root/'status.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError): qualification(root)
    def test_desired_board_has_pawn_on_e4_and_keeps_live_images_separate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);goal=save_goal(root,chess.STARTING_FEN,'e2e4')
            board=chess.Board(goal['desired_fen'])
            self.assertIsNone(board.piece_at(chess.E2));self.assertEqual(board.piece_at(chess.E4),chess.Piece(chess.PAWN,chess.WHITE))
            self.assertTrue((root/'desired_board.svg').is_file());self.assertFalse((root/'laptop_rgb.png').exists())
            with self.assertRaises(ValueError): save_goal(root,chess.STARTING_FEN,'e2e5')


if __name__=='__main__': unittest.main()
