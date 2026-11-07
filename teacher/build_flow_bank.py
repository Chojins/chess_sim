"""Reach consistent reset families with the analytic oracle, never learned credit.

All checkpoints are saved from real controlled physics with original board
baselines. No piece attachment, pose override, or task-state observation is used
by the oracle. This replaces mismatched legacy grasp poses for a new experiment.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from teacher.grasp_checkpoints import file_hash
from teacher.move_teacher import MoveConfig
from teacher.reactive_oracle import ReactiveOracle
from teacher.task_space import TaskSpaceMoveEnv
from teacher.train_grasp import save_status
from teacher.transport import StateClient
from teacher.verify_task_space import audit_trial


def families_now(env,score):
    t=env.task;a=env.base.actual
    center,height,error,contact=t.measure(a,env.base.encoder.point(a),env.base.command)
    fraction=float((center[1]-t.original[1])/(t.target[1]-t.original[1]))
    fingers=t.fingers(a);open_jaw=fingers[0]>4.8 and fingers[1]<-4.8
    return {
        'approach':not t.lifted and open_jaw and .03<error<.04,
        'open_grasp':not t.lifted and open_jaw and error<.003,
        'held_source':t.lifted and contact and .007<height<.016 and fraction<.2,
        'lift_clear':t.lifted and contact and height>=.027 and fraction<.5,
        'carry_mid':contact and height>.025 and .4<=fraction<=.7,
        'near_release':contact and np.linalg.norm(center[:2]-t.target[:2])<.004 and .002<height<.005,
        'retreat':t.released and open_jaw,
        'near_home':t.released and 6<score['home_error_deg']<12,
        'home_settle':t.released and score['home_error_deg']<3.5 and not t.done,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name-prefix',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8081')
    p.add_argument('--train-seeds',type=int,nargs='+',default=[101001,101002,101003])
    p.add_argument('--validation-seeds',type=int,nargs='+',default=[102001])
    p.add_argument('--test-seeds',type=int,nargs='+',default=[103001])
    args=p.parse_args();partitions={'train':args.train_seeds,'validation':args.validation_seeds,'test':args.test_seeds}
    seeds=sum(partitions.values(),[])
    if not all(partitions.values()) or len(set(seeds))!=len(seeds): p.error('Need disjoint nonempty seed groups')
    args.output.mkdir(parents=True,exist_ok=False);env=None;client=StateClient(args.url);entries=[];trials=[]
    bank_path=args.output/'manifest.json'
    base_bank={'format':'analytic_flow_move_reset_set_v1','case':{'id':'initial_e2e4','history':[],'move':'e2e4'},
        'entries':entries,'partitions':partitions,'reset_builder':'analytic_geometry_oracle',
        'learned_policy':False,'ready_for_act_collection':False,
        'limitations':'Physical reference/reset generation only; not a learned success. PhysX contact caches rebuild.'}
    def save(seed,split,family):
        t=env.task
        held=family in ('held_source','lift_clear','carry_mid','near_release')
        history={'move':'e2e4','seed':seed,'split':split,'family':family,'held_start':held,
            'prior_lift_verified':bool(t.lifted),'prior_max_lift_m':t.max_lift,
            'prior_release_verified':bool(t.released),'release_intent':t.release_location,
            'reset_builder':'analytic_geometry_oracle'}
        saved=env.base.client.save_reset(f'{args.name_prefix}_s{seed}_{family}',history)
        env.base.latest=saved
        entry={**saved['reset_context'],**history,'physical_step':t.steps,'prefix_step':t.steps}
        entries.append(entry);return entry
    try:
        for split,partition in partitions.items():
            for seed in partition:
                offset=np.random.default_rng(seed).uniform(-.5,.5,2)
                client.reset({'history':[],'move':'e2e4','pawn_offset_mm':offset.tolist()})
                history={'move':'e2e4','seed':seed,'split':split,'family':'home','held_start':False,
                    'prior_lift_verified':False,'prior_max_lift_m':0.,'prior_release_verified':False,'release_intent':None,
                    'reset_builder':'analytic_geometry_oracle','pawn_offset_mm':offset.tolist()}
                saved=client.save_reset(f'{args.name_prefix}_s{seed}_home',history)['reset_context']
                entry={**saved,**history,'physical_step':0,'prefix_step':0,'horizon':800};entries.append(entry)
                bank_path.write_text(json.dumps(base_bank,indent=2))
                if env is None: env=TaskSpaceMoveEnv(bank_path,args.url,split='train',config=MoveConfig(),adaptive=False,gamma=.997)
                env.fixed_entry=entry;obs,initial=env.reset(seed=seed);oracle=ReactiveOracle(env.cartesian_config)
                directory=args.output/'trials'/entry['name'];directory.mkdir(parents=True)
                (directory/'initial.json').write_text(json.dumps(initial,indent=2));saved_families={'home'};rows=[]
                with (directory/'steps.jsonl').open('x') as stream:
                    while True:
                        action,_=oracle.predict(obs);obs,reward,done,truncated,info=env.step(action)
                        row={'step':len(rows)+1,'action':action.tolist(),'reward':reward,'terminated':done,'truncated':truncated,
                            'score':info['score'],'teacher':info['teacher'],'reward_components':info['reward_components'],
                            'observation':info['telemetry']['observation']}
                        rows.append(row);stream.write(json.dumps(row)+'\n')
                        if not done and not truncated:
                            for family,ready in families_now(env,info['score']).items():
                                if ready and family not in saved_families:
                                    save(seed,split,family);saved_families.add(family)
                        if done or truncated: break
                result={'reset_name':entry['name'],'reset_family':'home','seed':seed,'steps':len(rows),'score':rows[-1]['score']}
                (directory/'result.json').write_text(json.dumps(result,indent=2));trials.append(result)
                if not result['score']['success'] or len(saved_families)!=10:
                    raise ValueError(f'Incomplete analytic reference seed {seed}: {result["score"]}, families {saved_families}')
                for e in entries:
                    if e['seed']==seed:
                        e['horizon']=800 if e['family']=='home' else min(800,max(96,int(np.ceil((len(rows)-e['physical_step'])*1.7+64))))
                save_status(args.output/'status.json',stage='building_analytic_flow_resets',completed_seeds=len(trials),checkpoints=len(entries))
                print('FLOW_REFERENCE',json.dumps(result),flush=True)
        bank_path.write_text(json.dumps(base_bank,indent=2))
        for entry in entries:
            ctx=client.restore_reset(entry['name'])['reset_context']
            if ctx['checkpoint_sha256']!=entry['sha256'] or max(ctx['parity'].values())>1e-5: raise ValueError('Checkpoint parity failed')
            entry['restore_parity']=ctx['parity']
        bank_path.write_text(json.dumps(base_bank,indent=2))
        manifest={'objective':'unified_e2e4','bank':str(bank_path.resolve()),'bank_sha256':file_hash(bank_path),'case':base_bank['case'],
            'action_config':asdict(env.base.config),'move_config':asdict(env.move_config),'joint_map':env.base.mapping,
            'observation_schema':env.schema,'observation_size':76,'policy_action_space':env.policy_action_space,
            'cartesian_config':asdict(env.cartesian_config),'oracle_config':asdict(oracle.config),
            'oracle_version':oracle.version,'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')},
            'learned_policy':False,'teacher_label':'analytic_geometry_oracle','ready_for_act_collection':False}
        (args.output/'policy_manifest.json').write_text(json.dumps(manifest,indent=2))
        checks=[audit_trial(args.output/'trials'/r['reset_name'],manifest) for r in trials]
        (args.output/'integrity.json').write_text(json.dumps({'passed':True,'results':checks},indent=2))
        save_status(args.output/'status.json',stage='analytic_flow_bank_ready',checkpoints=len(entries),
            completed_home_references=len(trials),learned_policy=False,ready_for_act_collection=False)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()
        client.close()


if __name__=='__main__': main()
