"""Plot saved USD/FK geometry without loading the simulator or using a GPU."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path('outputs/capture-grasp-audit-v7')
    report = json.loads((output/'report.json').read_text())
    geometry = np.load(output/'geometry.npz')
    labels = list(report['trials'])
    fig, axes = plt.subplots(2, len(labels), figsize=(17, 8), constrained_layout=True)
    for column, label in enumerate(labels):
        target = geometry[f'{label}_target']
        references = geometry[f'{label}_reference']
        for row, coordinates in enumerate(((1, 2), (0, 1))):
            ax = axes[row, column]
            pawn = (geometry[f'{label}_pawn']-target)*1000
            ax.scatter(pawn[:,coordinates[0]], pawn[:,coordinates[1]], s=.5, c='gray', alpha=.2, rasterized=True)
            for jaw, color in [('jaw_1','C0'),('jaw_2','C1')]:
                vertices = (geometry[f'{label}_{jaw}']-target)*1000
                ax.scatter(vertices[:,coordinates[0]],vertices[:,coordinates[1]],s=.7,c=color,alpha=.4,rasterized=True)
            refs = (references-target)*1000
            ax.scatter([0],[0],c='red',marker='o',s=35,label='Pawn target (17mm height)')
            ax.scatter([refs[0,coordinates[0]]],[refs[0,coordinates[1]]],c='black',marker='x',s=45,label='Fixed reference')
            ax.scatter([refs[1,coordinates[0]]],[refs[1,coordinates[1]]],c='purple',marker='+',s=45,label='Paired surface midpoint')
            ax.set_aspect('equal');ax.grid(alpha=.25)
            if row == 0:
                ax.set(xlim=(-30,30),ylim=(-25,35),xlabel='Front/back relative to target (mm)',ylabel='Height relative to target (mm)')
                sample = report['trials'][label]['closing_command_sample']
                ax.set_title(label.replace('_',' ')+'\nstep '+str(sample['step']),fontsize=10)
            else:
                ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='Left/right relative to target (mm)',ylabel='Front/back relative to target (mm)')
    axes[1,0].legend(fontsize=7,loc='upper left')
    fig.suptitle('Recorded closing poses: actual finger meshes and upright pawn approximation\n'
                 'Jaw 2 angle assumed opposite jaw 1; visual geometry does not establish physical contacts',fontsize=12)
    fig.savefig(output/'closing_poses.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
