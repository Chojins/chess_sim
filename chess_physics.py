"""Session-only contact geometry and piece measurements for manipulation."""
import numpy as np


def configure_contacts(stage, friction=1.0, piece_mass=0.005, board_friction=None, robot_self_collisions=False, robot_collision_geometry='hulls'):
    if robot_collision_geometry not in ('hulls','decomposed'): raise ValueError('Unknown robot collision geometry')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    material = UsdShade.Material.Define(stage, "/World/ChessContactMaterial")
    properties = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    properties.CreateStaticFrictionAttr(friction)
    properties.CreateDynamicFrictionAttr(friction)
    properties.CreateRestitutionAttr(0.0)

    def collider(prim, approximation=None):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if approximation:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(0.0005)
        px.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    squares = [bbox.ComputeWorldBound(p).ComputeAlignedRange()
               for p in Usd.PrimRange(stage.GetPrimAtPath('/World/board_assembly'))
               if p.GetName().startswith('tn__boardsquare_')]
    lo = np.min([r.GetMin() for r in squares], axis=0)
    hi = np.max([r.GetMax() for r in squares], axis=0)
    board = UsdGeom.Cube.Define(stage, "/World/ChessBoardCollider")
    board.CreateSizeAttr(1)
    board.AddTranslateOp().Set(Gf.Vec3d(*((hi + lo) / 2)))
    board.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    board.CreateVisibilityAttr("invisible")
    collider(board.GetPrim())
    if board_friction is not None:
        board_material = UsdShade.Material.Define(stage, '/World/ChessBoardMaterial')
        properties = UsdPhysics.MaterialAPI.Apply(board_material.GetPrim())
        properties.CreateStaticFrictionAttr(board_friction)
        properties.CreateDynamicFrictionAttr(board_friction)
        properties.CreateRestitutionAttr(0.0)
        # Use the board coefficient for piece/board contact while preserving
        # the higher coefficient at the gripper fingers.
        PhysxSchema.PhysxMaterialAPI.Apply(board_material.GetPrim()).CreateFrictionCombineModeAttr('min')
        UsdShade.MaterialBindingAPI.Apply(board.GetPrim()).Bind(board_material, materialPurpose='physics')
    from grasp_geometry import piece_grasp_target
    paths, centers, robot_colliders = [], [], {}
    self_filters = []
    disabled_duplicates=[]
    grasp_points = []
    cache = UsdGeom.XformCache()
    for color in ('blue', 'red'):
        group = stage.GetPrimAtPath('/World/chess_pieces_' + color)
        for prim in group.GetChildren():
            meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
            if not meshes:
                continue
            center = bbox.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
            matrix = cache.GetLocalToWorldTransform(prim)
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            grasp = piece_grasp_target(str(prim.GetPath()), bounds.GetMin(), bounds.GetMax())
            grasp_points.append((np.array(matrix.GetInverse().Transform(Gf.Vec3d(*map(float, grasp)))) *
                                 np.array(Gf.Transform(matrix).GetScale())).tolist())
            # RigidPrim world poses omit scale; imported piece meshes use mm.
            centers.append((np.array(matrix.GetInverse().Transform(center)) *
                            np.array(Gf.Transform(matrix).GetScale())).tolist())
            paths.append(str(prim.GetPath()))
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateRigidBodyEnabledAttr(True)
            body.CreateVelocityAttr(Gf.Vec3f(0))
            body.CreateAngularVelocityAttr(Gf.Vec3f(0))
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(piece_mass)
            px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            px.CreateSolverPositionIterationCountAttr(16)
            px.CreateSolverVelocityIterationCountAttr(4)
            for mesh in meshes:
                collider(mesh, 'convexHull')
        root = stage.GetPrimAtPath('/World/so100_' + color)
        # Some visual groups (including the blue left jaw) are instances.
        # Ordinary USD traversal skips their children, and instance proxies
        # cannot receive per-instance collision opinions. Expand only the
        # robot visuals in the session layer before adding colliders.
        while True:
            instances = [p for p in Usd.PrimRange(root) if p.IsInstance() and (
                '/visuals' in str(p.GetPath()) or (robot_collision_geometry=='decomposed' and '/collisions' in str(p.GetPath())))]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        # Coarse housing hulls and meshing jaw gears overlap internally. Keep
        # external contacts enabled; self-contact needs dedicated CAD colliders.
        PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(str(root.GetPath()) + '/root_joint')).CreateEnabledSelfCollisionsAttr(robot_self_collisions)
        if robot_self_collisions:
            # Exclude mechanically connected neighbours and the two meshing
            # jaw gears. Non-adjacent housing pairs remain collidable, including
            # gripper/camera housing against the shoulder's joint-2 servo.
            pairs=[]
            for joint in Usd.PrimRange(root):
                if not joint.IsA(UsdPhysics.Joint): continue
                a=UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()
                b=UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()
                if len(a)==len(b)==1: pairs.append((a[0],b[0]))
            pairs.append((root.GetPath().AppendChild('jaw_1'),root.GetPath().AppendChild('jaw_2')))
            for a,b in pairs:
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(b)
                self_filters.append([str(a),str(b)])
            for body in Usd.PrimRange(root):
                if body.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) and '/visuals/' in str(prim.GetPath()):
                printed_housing=robot_collision_geometry=='decomposed' and '_Motor/' not in str(prim.GetPath())
                approximation='convexDecomposition' if '/jaw_' in str(prim.GetPath()) or printed_housing else 'convexHull'
                collider(prim,approximation)
                robot_colliders[str(prim.GetPath())]=approximation
        if robot_collision_geometry=='decomposed':
            # The imported arm already has duplicate housing colliders. A coarse
            # original hull would fill the voids in a decomposed replacement.
            for prim in Usd.PrimRange(root):
                path=str(prim.GetPath())
                if not prim.IsA(UsdGeom.Mesh) or '/collisions/' not in path: continue
                replacement=path.replace('/collisions/','/visuals/')
                if replacement not in robot_colliders:
                    raise ValueError(f'No visual collision replacement for {path}')
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
                disabled_duplicates.append({'disabled':path,'replacement':replacement})
        for jaw in ('jaw_1', 'jaw_2'):
            prefix = str(root.GetPath()) + '/' + jaw + '/'
            if not any(path.startswith(prefix) for path in robot_colliders):
                raise RuntimeError(f'Missing collision mesh for {color} {jaw}')
    result={'piece_grasp_local_m': grasp_points, 'piece_mass_kg': piece_mass, 'friction': friction,
        'board_frame_bounds_m': [list(bbox.ComputeWorldBound(stage.GetPrimAtPath('/World/board_assembly')).ComputeAlignedRange().GetMin()),
                                 list(bbox.ComputeWorldBound(stage.GetPrimAtPath('/World/board_assembly')).ComputeAlignedRange().GetMax())],
        'square_centers_m': [list(r.GetMidpoint()) for r in squares],
        'board_top_m': float(hi[2]), 'board_friction': friction if board_friction is None else board_friction,
        'piece_collision': 'convexHull',
        'jaw_collision': 'convexDecomposition', 'contact_offset_m': 0.0005,
        'robot_self_collisions': robot_self_collisions, 'robot_collision_meshes': robot_colliders}
    if robot_self_collisions:
        result.update(robot_self_collision_mode='non_adjacent_with_meshing_jaw_pair_excluded_v1',
            robot_self_collision_filtered_pairs=self_filters,robot_contact_reporting=True,
            collision_geometry_scope='Existing housing convex hulls; experimental conservative self-contact, not calibrated clearance')
    if robot_collision_geometry=='decomposed':
        result.update(robot_collision_geometry='decomposed_printed_housings_deduplicated_v1',
            disabled_duplicate_robot_colliders=disabled_duplicates,
            collision_geometry_scope='Convex-decomposed printed housings/jaws, convex servo hulls; imported duplicate hulls disabled')
    return paths, np.asarray(centers), result


class PieceMonitor:
    def __init__(self, view, paths, local_centers, local_grasp_points=None):
        self.view, self.paths, self.local_centers = view, paths, local_centers
        self.local_grasp_points = local_grasp_points
        self.initial_poses = tuple(x.copy() for x in view.get_world_poses())
        self.initial_centers = self.centers()
        self.maximum_lift = np.zeros(len(paths))
        self.active_index = paths.index('/World/chess_pieces_blue/pawn_03')

    def centers(self):
        return self.world_points(self.local_centers)

    def world_points(self, local_points):
        from pxr import Gf
        positions, rotations = self.view.get_world_poses()
        return np.array([np.asarray(Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))).TransformDir(
            Gf.Vec3d(*map(float, center)))) + pos for pos, q, center in zip(positions, rotations, local_points)])

    def reset(self):
        self.view.set_world_poses(*self.initial_poses)
        self.view.set_velocities(np.zeros((len(self.paths), 6)))
        self.maximum_lift[:] = 0

    def sample(self):
        centers = self.centers()
        if not np.isfinite(centers).all():
            raise RuntimeError('Non-finite piece pose')
        self.maximum_lift = np.maximum(self.maximum_lift, centers[:, 2] - self.initial_centers[:, 2])
        index = self.active_index
        target = self.initial_centers[index].copy()
        target[1] += 0.045
        from pxr import Gf
        rotation = self.view.get_world_poses()[1][index]
        initial_rotation = self.initial_poses[1][index]
        def up(q):
            return np.array(Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))).TransformDir(Gf.Vec3d(0, 0, 1)))
        upright = float(np.dot(up(rotation), up(initial_rotation)))
        rotations = self.view.get_world_poses()[1]
        uprights = [float(np.dot(up(q), up(q0))) for q, q0 in zip(rotations, self.initial_poses[1])]
        moved = np.linalg.norm(centers[:, :2] - self.initial_centers[:, :2], axis=1)
        moved[index] = 0
        grasp = {} if self.local_grasp_points is None else {
            'piece_grasp_targets_m': self.world_points(self.local_grasp_points).tolist()}
        return {**grasp, 'pawn_center_m': centers[index].tolist(), 'pawn_lift_max_m': float(self.maximum_lift[index]),
                'pawn_target_xy_error_m': float(np.linalg.norm(centers[index, :2] - target[:2])),
                'other_pieces_displaced_over_3mm': int((moved > 0.003).sum()),
                'pawn_upright_cosine': upright,
                'pawn_height_change_m': float(centers[index, 2] - self.initial_centers[index, 2]),
                'piece_centers_m': centers.tolist(), 'piece_paths': self.paths,
                'piece_upright_cosines': uprights}
