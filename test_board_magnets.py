"""Numerical checks; run with Python + NumPy, without Isaac."""
import unittest

import numpy as np

from board_magnets import MagnetSettings, attraction


class MagnetForceTests(unittest.TestCase):
    def test_force_matches_potential_gradient(self):
        settings = MagnetSettings()
        squares = np.array([[0., 0., 0.], [.0225, 0., 0.]])
        point = np.array([.003, .004, .002])

        def energy(point):
            distance = point - squares
            return -settings.energy_j * np.exp(
                -np.sum(distance[:, :2] ** 2, axis=1) / (2 * settings.lateral_width_m ** 2)
                -distance[:, 2] / settings.height_decay_m).sum()

        gradient = []
        for step in np.eye(3) * 1e-7:
            gradient.append(-(energy(point + step) - energy(point - step)) / 2e-7)
        force = attraction(point[None], np.zeros((1, 3)), squares, settings)[0]
        np.testing.assert_allclose(force, gradient, rtol=1e-7)

    def test_damping_removes_energy(self):
        settings = MagnetSettings()
        points = np.array([[.003, .004, .002], [-.002, .001, .005]])
        velocity = np.array([[.1, -.2, .3], [-.2, .1, -.3]])
        squares = np.zeros((1, 3))
        damping = (attraction(points, velocity, squares, settings) -
                   attraction(points, np.zeros_like(velocity), squares, settings))
        self.assertTrue(np.all(np.sum(damping * velocity, axis=1) < 0))

    def test_boundary_balance_and_height_decay(self):
        settings = MagnetSettings()
        squares = np.array([[-.01125, 0, 0], [.01125, 0, 0]])
        balanced = attraction(np.zeros((1, 3)), np.zeros((1, 3)), squares, settings)
        self.assertAlmostEqual(balanced[0, 0], 0)
        points = np.array([[0., 0., 0.], [0., 0., .02]])
        force = attraction(points, np.zeros_like(points), np.zeros((1, 3)), settings)
        self.assertAlmostEqual(force[0, 2], -.14)
        self.assertLess(abs(force[1, 2] / force[0, 2]), .01)


if __name__ == '__main__':
    unittest.main()
