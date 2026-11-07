"""Offline finger/pose audit of recorded pickups; never advances physics.

Run with ./run_usd_python.sh diagnose_capture_grasp.py. The fits are
kinematic feasibility checks, not collision-free or stable-grasp guarantees.
"""
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.optimize import least_squares

from episode_replay import EpisodeReplay
from grasp_geometry import GRASP_LOCAL_M


TRIALS = {
    'e2e4_success': ('outputs/opening-recorded-v6/repeat_0/initial_e2e4/learned_1102', 126),
    'capture_success': ('outputs/capture-pickup-only-v7/scandinavian_capture/before_1202', 130),
    'capture_miss_high': ('outputs/capture-pickup-only-v7/scandinavian_capture/before_1203', 138),
    'capture_miss_final': ('outputs/capture-pickup-only-v7/scandinavian_capture/learned_1201', 134),
    'capture_miss_back': ('outputs/opening-recorded-v6/repeat_0/scandinavian_capture/learned_1102', 111),
}


def transform(points, matrix):
    """USD row-vector convention, including translation."""
    return np.c_[points, np.ones(len(points))].dot(np.asarray(matrix))[:, :3]


def mesh_in_body(stage, body):
    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(body)
    meshes = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f'Expected one finger mesh in {body}')
    mesh = UsdGeom.Mesh(meshes[0])
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError('Expected triangulated finger mesh')
    matrix = cache.GetLocalToWorldTransform(meshes[0]) * cache.GetLocalToWorldTransform(root).GetInverse()
    return transform(np.asarray(mesh.GetPointsAttr().Get()), matrix), np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)


def section(triangles, axis, coordinate):
    """Triangle/plane intersections, omitting triangles coplanar with plane."""
    lines = []
    for triangle in triangles:
        points = []
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            if (a[axis] < coordinate <= b[axis]) or (b[axis] < coordinate <= a[axis]):
                points.append(a + (b - a) * ((coordinate - a[axis]) / (b[axis] - a[axis])))
        if len(points) == 2:
            lines.append(points)
    return np.asarray(lines)


def main():
    output = Path('outputs/capture-grasp-audit-v7')
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open('Sim_Stage.usd')
    stage.SetEditTarget(stage.GetSessionLayer())
    fk = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                       'calibration/legacy_blue_joint_map_physics.json', output, kinematic=False)
    root = fk.root
    jaw_names = ('jaw_1', 'jaw_2')
    meshes = [mesh_in_body(stage, f'{root}/{jaw}') for jaw in jaw_names]

    def matrices(q):
        angles = dict(zip(fk.mapping.sim_names, q))
        angles['jaw2_hinge'] = -float(q[5])
        return fk.forward(angles)

    # Derive opposing surface points by intersecting the real mesh at the
    # user's axial setback and local Z=0. These are material points, not an
    # independently identified centre of the printed V groove.
    zero = np.zeros(6)
    zero_matrices = matrices(zero)
    inverse = zero_matrices[f'{root}/gripper'].GetInverse()
    anchors, cross_sections = [], {}
    for name, (vertices, faces) in zip(jaw_names, meshes):
        into_gripper = zero_matrices[f'{root}/{name}'] * inverse
        v = transform(vertices, into_gripper)
        lines = section(v[faces], 2, 0.)
        crossings = []
        y = GRASP_LOCAL_M[1]
        for a, b in lines:
            if (a[1] < y <= b[1]) or (b[1] < y <= a[1]):
                crossings.append(a + (b-a)*((y-a[1])/(b[1]-a[1])))
        if not crossings:
            raise ValueError('No finger surface at requested setback')
        contact = min(crossings, key=lambda p: abs(p[0]))
        anchors.append(transform(contact[None], into_gripper.GetInverse())[0])
        cross_sections[name] = lines

    def points(q):
        m = matrices(q)
        tool = m[f'{root}/gripper']
        static = transform(np.asarray(GRASP_LOCAL_M)[None], tool)[0]
        contacts = np.array([transform(a[None], m[f'{root}/{name}'])[0]
                             for a, name in zip(anchors, jaw_names)])
        return static, contacts.mean(0), contacts, tool

    audit = {'note': 'Offline visual-mesh/FK audit. Jaw 2 assumed opposite measured jaw 1; '
                     'actual follower angle, contact forces and PhysX convex hulls were not logged. '
                     'Closing-command samples are not measured first-contact events.',
             'surface_anchor_definition': 'Opposing mesh surfaces at zero hinge angle, user setback Y, local Z=0',
             'surface_anchors_jaw_local_m': [a.tolist() for a in anchors],
             'trials': {}, 'opening_sweep': []}
    plot_arrays = dict(cross_sections)
    reference_path, reference_step = TRIALS['e2e4_success']
    reference_row = json.loads((Path(reference_path)/'steps.jsonl').read_text().splitlines()[reference_step-1])
    reference_q = reference_row['placement_telemetry']['measured_joint_deg']
    reference_tool = points(reference_q)[3]
    reference_pitch = float(np.arccos(np.clip(reference_tool.TransformDir(Gf.Vec3d(0, 1, 0))[2], -1, 1)))
    reference_span = float(np.arcsin(np.clip(reference_tool.TransformDir(Gf.Vec3d(1, 0, 0))[2], -1, 1)))

    for label, (directory, closing_step) in TRIALS.items():
        path = Path(directory)
        initial = json.loads((path/'initial.json').read_text())['telemetry']['observation']
        geometry = initial['task_geometry']
        index = geometry['capture_index'] if geometry['capture_index'] is not None else geometry['source_index']
        rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
        samples = []
        # Include approach, closure and initial lift, rather than only one frame.
        for row in rows[max(0, closing_step-25):closing_step+25]:
            t = row['placement_telemetry']
            q = np.asarray(t['measured_joint_deg'])
            static, moving, contacts, tool = points(q)
            target = np.asarray(t['pieces']['piece_grasp_targets_m'][index])
            mismatch = np.linalg.norm(static-t['gripper_grasp_center_m'])
            if mismatch > 1e-6:
                raise AssertionError(f'FK differs from telemetry by {mismatch} m')
            samples.append({'step': row['step'], 'measured_joint_deg': q.tolist(),
                            'commanded_joint_deg': fk.mapping.to_sim_deg(row['residual']['executed_legacy']).tolist(),
                            'target_m': target.tolist(), 'fixed_reference_m': static.tolist(),
                            'surface_midpoint_m': moving.tolist(),
                            'fixed_minus_target_mm': ((static-target)*1000).tolist(),
                            'midpoint_minus_target_mm': ((moving-target)*1000).tolist(),
                            'midpoint_minus_fixed_mm': ((moving-static)*1000).tolist(),
                            'surface_anchor_span_mm': float(np.linalg.norm(contacts[1]-contacts[0])*1000),
                            'approach_tilt_deg': float(np.degrees(np.arccos(np.clip(tool.TransformDir(Gf.Vec3d(0,1,0))[2],-1,1)))),
                            'span_tilt_deg': float(np.degrees(np.arcsin(np.clip(tool.TransformDir(Gf.Vec3d(1,0,0))[2],-1,1))))})
        sample = next(s for s in samples if s['step'] == closing_step)
        row = rows[closing_step-1]
        q = np.array(sample['measured_joint_deg'])
        target = np.array(sample['target_m'])
        commanded = np.array(sample['commanded_joint_deg'])
        base = commanded - np.array(row['learning']['applied_residual'])*np.array([2,2,2,2,2,3])
        limits = np.array(initial['joint_limits_deg'])
        lower = np.maximum(-2., limits[:5,0]-base[:5])
        upper = np.minimum(2., limits[:5,1]-base[:5])
        fits = {}
        for mode in ('fixed_position', 'surface_midpoint_position', 'surface_midpoint_and_orientation'):
            def errors(delta):
                candidate = base.copy()
                candidate[:5] += delta
                # Fit at observed aperture. Finger motor contact/tracking is
                # outside this static feasibility check.
                candidate[5] = q[5]
                static, moving, _, tool = points(candidate)
                position = static if mode == 'fixed_position' else moving
                result = (position-target)*1000
                if mode.endswith('_and_orientation'):
                    # Preserve successful e2e4 tilt while allowing yaw toward
                    # another square; 50mm is an explicit orientation weight.
                    pitch = np.arccos(np.clip(tool.TransformDir(Gf.Vec3d(0,1,0))[2], -1, 1))
                    span = np.arcsin(np.clip(tool.TransformDir(Gf.Vec3d(1,0,0))[2], -1, 1))
                    result = np.r_[result, 50*(pitch-reference_pitch), 50*(span-reference_span)]
                return result
            fit = least_squares(errors, np.zeros(5).clip(lower,upper), bounds=(lower,upper),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11)
            candidate = base.copy();candidate[:5] += fit.x;candidate[5] = q[5]
            static, moving, _, tool = points(candidate)
            fits[mode] = {'residual_deg': fit.x.tolist(), 'position_error_mm': float(np.linalg.norm(errors(fit.x)[:3])),
                          'measured_to_fit_delta_deg': (candidate[:5]-q[:5]).tolist(),
                          'grasp_candidate_joint_deg': candidate.tolist(),
                          'approach_tilt_deg': float(np.degrees(np.arccos(np.clip(tool.TransformDir(Gf.Vec3d(0,1,0))[2],-1,1)))),
                          'span_tilt_deg': float(np.degrees(np.arcsin(np.clip(tool.TransformDir(Gf.Vec3d(1,0,0))[2],-1,1))))}
        range_fits = {}
        for bound in (2., 4., 6.):
            lo = np.maximum(-bound, limits[:5,0]-base[:5])
            hi = np.minimum(bound, limits[:5,1]-base[:5])
            def position_error(delta):
                candidate = base.copy();candidate[:5] += delta;candidate[5] = q[5]
                return (points(candidate)[0]-target)*1000
            fit = least_squares(position_error, np.zeros(5).clip(lo,hi), bounds=(lo,hi),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11)
            range_fits[str(bound)] = {'residual_deg': fit.x.tolist(),
                                     'fixed_reference_error_mm': float(np.linalg.norm(position_error(fit.x)))}
        audit['trials'][label] = {'source': directory, 'closing_command_sample': sample, 'bounded_fits': fits,
                                 'tracking_error_deg': (q-commanded).tolist(), 'range_fits': range_fits}
        (output/f'{label}.json').write_text(json.dumps(samples, indent=2))
        # Opening plane projected in world YZ makes front/back and height
        # differences visible. Save meshes for plotting without USD imports.
        m = matrices(q)
        for name, (vertices, faces) in zip(jaw_names, meshes):
            plot_arrays[f'{label}_{name}'] = transform(vertices, m[f'{root}/{name}'])
        piece_path = geometry['capture_path'] if geometry['capture_index'] is not None else geometry['source_path']
        cache = UsdGeom.XformCache()
        piece_vertices = []
        for p in Usd.PrimRange(stage.GetPrimAtPath(piece_path), Usd.TraverseInstanceProxies()):
            if p.IsA(UsdGeom.Mesh):
                piece_vertices.extend(transform(np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()),
                                                cache.GetLocalToWorldTransform(p)))
        piece_vertices = np.asarray(piece_vertices)
        authored_center = (piece_vertices.min(0)+piece_vertices.max(0))/2
        observed_center = np.asarray(row['placement_telemetry']['pieces']['piece_centers_m'][index])
        # Full piece quaternions were not recorded; closing samples are
        # effectively upright. This translated authored shape is a visual
        # approximation, not a contact or penetration measurement.
        if row['placement_telemetry']['pieces']['piece_upright_cosines'][index] < .999:
            raise ValueError('Cannot display tilted piece without full orientation')
        plot_arrays[f'{label}_pawn'] = piece_vertices-authored_center+observed_center
        plot_arrays[f'{label}_target'] = target
        plot_arrays[f'{label}_reference'] = np.array([sample['fixed_reference_m'], sample['surface_midpoint_m']])

    for angle in (-3, 0, 4, 10, 20, 30, 45):
        q = np.zeros(6);q[5] = angle
        static, moving, contacts, tool = points(q)
        local_shift = tool.GetInverse().TransformDir(Gf.Vec3d(*map(float,moving-static)))
        audit['opening_sweep'].append({'hinge_deg': angle, 'midpoint_shift_gripper_local_mm': (np.array(local_shift)*1000).tolist(),
                                        'surface_anchor_span_mm': float(np.linalg.norm(contacts[1]-contacts[0])*1000)})
    np.savez_compressed(output/'geometry.npz', **plot_arrays)
    (output/'report.json').write_text(json.dumps(audit, indent=2))
    for label, entry in audit['trials'].items():
        print(label, json.dumps({'fixed_error_mm':entry['closing_command_sample']['fixed_minus_target_mm'],
                                'midpoint_error_mm':entry['closing_command_sample']['midpoint_minus_target_mm'],
                                'fits':entry['bounded_fits']}))
    print('opening_sweep', json.dumps(audit['opening_sweep']))


if __name__ == '__main__':
    main()
