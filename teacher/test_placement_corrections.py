import unittest
import numpy as np
from teacher.placement_corrections import placement_mask


class PlacementCorrectionTests(unittest.TestCase):
    def test_only_obstructed_held_pieces_above_board_near_goal_are_corrected(self):
        x=np.zeros((6,76),np.float32)
        x[:,5]=-3/180;x[:,21]=.2;x[:,22]=-.2;x[:,17]=.012/.03;x[:,19]=.015/.1
        x[1,21]=.6  # Open jaw.
        x[2,17]=0  # Grasp still resting on the board.
        x[3,19]=.05/.1  # Not the placement region.
        x[4,21]=-.3;x[4,22]=.3  # Fully closed, empty gripper.
        x[5,12]=.02/.03  # Material grasp is outside the opening.
        np.testing.assert_array_equal(placement_mask(x),[True,False,False,False,False,False])


if __name__=='__main__':unittest.main()
