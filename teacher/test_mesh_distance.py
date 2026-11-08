import unittest
import numpy as np
from teacher.mesh_distance import point_mesh_distance


class MeshDistanceTests(unittest.TestCase):
    def test_triangle_interior_edge_and_vertex_distances(self):
        surface=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
        values=point_mesh_distance([[.2,.2,.003],[.3,.3,0],[.5,-.004,0],[-3,-4,0]],surface)
        np.testing.assert_allclose(values,[.003,0,.004,5],atol=1e-12)

    def test_degenerate_triangle_still_has_edge_distance(self):
        surface=np.array([[[0,0,0],[1,0,0],[1,0,0]],[[4,4,4],[4,4,4],[4,4,4]]],float)
        np.testing.assert_allclose(point_mesh_distance([[.5,.1,0],[4,4,4]],surface),[.1,0],atol=1e-12)


if __name__=='__main__':unittest.main()
