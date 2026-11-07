"""Export actual finger/pawn meshes and plot a measured, open grasp pose.

Use run_usd_python.sh for export, run_rl_python.sh for plotting. Only reads
the authored stage; the plot is a CAD inspection, not a camera rendering.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from grasp_geometry import GRASP_LOCAL_M, GRASP_REFERENCE


def joint_transforms(specification, q, jaw2=None):
    angles = dict(zip(specification['names'], q))
    angles['jaw2_hinge'] = -q[5] if jaw2 is None else jaw2
    result = {}
    for joint in specification['joints']:
        x, y, z = joint['axis']
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        angle = np.deg2rad(angles[joint['name']])
        rotation = np.eye(4)
        rotation[:3, :3] = (np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)).T
        parent = result.get(joint['parent'], np.array(joint['parent_world']))
        result[joint['child']] = np.array(joint['f1_inverse']) @ rotation @ joint['f0'] @ parent
    return result


def transform(points, matrix):
    return (np.c_[points, np.ones(len(points))] @ np.asarray(matrix))[:, :3]


def opening_center(specification, anchors, q, jaw2=None):
    frames = joint_transforms(specification, q, jaw2)
    root = specification['tool_link'].rsplit('/', 1)[0]
    points = [transform(np.asarray(anchor)[None], frames[f'{root}/jaw_{i+1}'])[0]
              for i, anchor in enumerate(anchors)]
    return np.mean(points, axis=0)


def export(output):
    from pxr import Usd, UsdGeom, Gf
    from tempfile import TemporaryDirectory
    from episode_replay import EpisodeReplay
    from diagnose_capture_grasp import mesh_in_body, section
    import hashlib
    # USD's interpreter does not share the learner's optional torch packages.
    stage = Usd.Stage.Open('Sim_Stage.usd')
    stage.SetEditTarget(stage.GetSessionLayer())
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='grasp-review-') as temporary:
        fk = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                           'calibration/legacy_blue_joint_map_physics.json', temporary, kinematic=False)
        zero = fk.forward(dict(zip(fk.mapping.sim_names + ['jaw2_hinge'], [0.]*7)))
        inverse = zero[f'{fk.root}/gripper'].GetInverse()
        geometry, anchors = {}, []
        for i in (1, 2):
            name = f'jaw_{i}'
            vertices, faces = mesh_in_body(stage, f'{fk.root}/{name}')
            matrix = zero[f'{fk.root}/{name}'] * inverse
            lines = section(transform(vertices, matrix)[faces], 2, 0.)
            crossings = []
            for a, b in lines:
                y = GRASP_LOCAL_M[1]
                if (a[1] < y <= b[1]) or (b[1] < y <= a[1]):
                    crossings.append(a + (b-a)*((y-a[1])/(b[1]-a[1])))
            if not crossings:
                raise ValueError('No opposing finger surface at the specified setback')
            contact = min(crossings, key=lambda p: abs(p[0]))
            anchors.append(transform(contact[None], matrix.GetInverse())[0].tolist())
            geometry[name+'_vertices'] = vertices
            geometry[name+'_faces'] = faces
        cache = UsdGeom.XformCache()
        pieces, faces, offset = [], [], 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/chess_pieces_red/pawn_03'), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            vertices = transform(np.asarray(mesh.GetPointsAttr().Get()), cache.GetLocalToWorldTransform(prim))
            if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
                raise ValueError('Expected triangulated pawn mesh')
            pieces.append(vertices)
            faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)+offset)
            offset += len(vertices)
        pawn = np.concatenate(pieces)
        geometry['pawn_vertices'] = pawn - (pawn.min(0)+pawn.max(0))/2
        geometry['pawn_faces'] = np.concatenate(faces)
        np.savez_compressed(output/'meshes.npz', **geometry)
        (output/'reference.json').write_text(json.dumps({'anchors_jaw_local_m': anchors,
            'grasp_reference': GRASP_REFERENCE, 'usd_sha256': hashlib.sha256(Path('Sim_Stage.usd').read_bytes()).hexdigest(),
            'definition': 'opposing mesh surfaces at 7 mm setback in zero-hinge geometry; midpoint follows both measured fingers'}, indent=2))
    print('Exported current USD finger meshes and opposing surface references')


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.collections import PolyCollection
    model = json.loads((output/'reference.json').read_text())
    spec = json.loads(Path('outputs/capture-supervised-kinematics-v10.json').read_text())
    report = json.loads((output/'pose.json').read_text())
    obs = json.loads((output/'final_observation.json').read_text())
    frames = joint_transforms(spec, obs['measured_joint_deg'], obs['measured_finger_joint_deg']['jaw2_hinge'])
    root = spec['tool_link'].rsplit('/', 1)[0]
    target = np.array(report['target_m'])
    with np.load(output/'meshes.npz') as meshes:
        shapes = []
        for i, color in ((1, '#2466bc'), (2, '#8353ac')):
            v = transform(meshes[f'jaw_{i}_vertices'], frames[f'{root}/jaw_{i}'])
            shapes.append(((v-target)*1000, meshes[f'jaw_{i}_faces'], color, f'Finger {i}'))
        pawn = meshes['pawn_vertices'] + obs['pieces']['piece_centers_m'][report['victim_index']]
        shapes.append(((pawn-target)*1000, meshes['pawn_faces'], '#d05c37', 'Pawn'))
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, (a, b), title in zip(axes, ((0, 2), (1, 2), (0, 1)),
                                      ('Front: across the opening', 'Side: depth and height', 'Top: forward/back centring')):
            for vertices, triangles, color, label in shapes:
                ax.add_collection(PolyCollection(vertices[triangles][:, :, [a, b]], facecolors=color,
                                                 edgecolors='none', alpha=.13, rasterized=True, label=label))
            center = (np.array(report['opening_center_m'])-target)*1000
            fixed = (np.array(obs['gripper_grasp_center_m'])-target)*1000
            ax.plot(0, 0, 'o', color='black', markersize=8, fillstyle='none', label='Requested grasp point')
            ax.plot(center[a], center[b], '+', color='#00a1a7', markersize=13, markeredgewidth=2, label='Measured opening midpoint')
            ax.plot(fixed[a], fixed[b], 'x', color='#af8010', markersize=8, label='Fixed zero-angle reference')
            if b == 2:
                ax.axhline(-17, color='#666', linestyle='--', linewidth=1, label='Pawn bottom')
            ax.set(xlim=(-35, 35), ylim=(-22, 48) if b == 2 else (-35, 35),
                   xlabel=f"World {'XYZ'[a]} from target (mm)", ylabel=f"World {'XYZ'[b]} from target (mm)", title=title)
            ax.set_aspect('equal'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7, loc='upper right')
        figure.suptitle('Measured open-gripper CAD inspection: target 17 mm above pawn bottom, 7 mm finger setback\n'
                       'Actual finger meshes; upright pawn mesh at its measured centre. This is not a contact test.', fontsize=11)
        figure.tight_layout()
        figure.savefig(output/'geometry_views.png', dpi=150)
        plt.close(figure)
    # Enlarged crops of this single inspection pose, using the saved raw
    # camera pixels. No synthetic camera view or pose annotation is inserted.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, bounds, title in zip(axes, ('red', 'blue'),
            ((270, 235, 395, 330), (180, 0, 465, 230)),
            ('Fixed camera: enlarged crop', 'Wrist camera: enlarged crop')):
        image = cv2.cvtColor(cv2.imread(str(output/f'{name}_camera.png')), cv2.COLOR_BGR2RGB)
        x0, y0, x1, y1 = bounds
        ax.imshow(image[y0:y1, x0:x1]); ax.set_title(title); ax.axis('off')
    figure.suptitle('Actual paused simulation frame: open gripper centred at the requested pawn grasp point')
    figure.tight_layout()
    figure.savefig(output/'camera_closeups.png', dpi=150)
    plt.close(figure)
    print('Saved geometry_views.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'plot'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (export if args.mode == 'export' else plot)(args.output)
