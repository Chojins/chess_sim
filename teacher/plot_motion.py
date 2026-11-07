"""Plot actual recorded movement and grip timing without rerunning physics."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite motion plot')
    initial=json.loads((args.trial/'initial.json').read_text())['telemetry']['observation']
    rows=[json.loads(s) for s in (args.trial/'steps.jsonl').read_text().splitlines()]
    actual=np.array([initial['measured_joint_deg']]+[r['observation']['measured_joint_deg'] for r in rows])
    command=np.array([r['teacher']['command_joint_deg'] for r in rows])
    time=np.arange(1,len(rows)+1)/30;velocity=np.diff(actual,axis=0)*30
    fingers=np.array([[r['observation']['measured_finger_joint_deg']['jaw1_hinge'],
        -r['observation']['measured_finger_joint_deg']['jaw2_hinge']] for r in rows])
    events={}
    for name,test in [('close command',lambda r:r['teacher']['command_joint_deg'][5]<0),
                      ('verified lift',lambda r:r['score']['lifted']),
                      ('released on target',lambda r:r['score']['released_on_target'])]:
        events[name]=next((r['step']/30 for r in rows if test(r)),None)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    axes[0].plot(time,np.max(np.abs(velocity[:,:5]),axis=1),color='#1766a6')
    axes[0].set_ylabel('Max actual arm speed\n(degrees / second)');axes[0].set_ylim(bottom=0)
    axes[1].plot(time,fingers[:,0],label='actual finger 1');axes[1].plot(time,fingers[:,1],label='actual finger 2 (mirrored)')
    axes[1].plot(time,command[:,5],ls='--',label='jaw command');axes[1].set_ylabel('Jaw angles (degrees)');axes[1].legend(loc='upper right')
    axes[2].plot(time,[r['score']['height_m']*1000 for r in rows],label='pawn height above starting surface')
    axes[2].plot(time,[r['score']['target_error_m']*1000 for r in rows],label='pawn distance from e4 centre')
    axes[2].set_ylabel('Distance (mm)');axes[2].set_xlabel('Simulation time (seconds)');axes[2].legend(loc='upper right')
    for ax in axes:
        ax.grid(alpha=.2)
        for name,when in events.items():
            if when is not None: ax.axvline(when,color='#777777',lw=.8,alpha=.6)
    for name,when in events.items():
        if when is not None: axes[0].text(when+.1,axes[0].get_ylim()[1]*.93,name,rotation=90,va='top',fontsize=8)
    fig.suptitle(f'{args.trial.parent.name}: {rows[-1]["score"]["reason"]}\nActual recorded control and physics; diagnostics do not certify generalization')
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=150);plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({'trial':str(args.trial.resolve()),'steps':len(rows),
        'simulated_seconds':len(rows)/30,'events_s':events,'score':rows[-1]['score'],
        'max_measured_arm_speed_deg_s':float(np.max(np.abs(velocity[:,:5]))),
        'rms_measured_arm_acceleration_deg_s2':float(np.sqrt(np.mean((np.diff(velocity[:,:5],axis=0)*30)**2)))},indent=2))


if __name__=='__main__': main()
