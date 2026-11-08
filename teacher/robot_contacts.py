"""Observe robot self/inter-robot contacts; never commands or masks contacts."""
import numpy as np


def robot_pair(a,b):
    def color(path):
        for value in ('blue','red'):
            if path.startswith('/World/so100_'+value+'/'): return value
        return None
    ca,cb=color(a),color(b)
    if ca is None or cb is None: return None
    return 'self' if ca==cb else 'inter_robot'


def audit_collision_trace(initial,rows):
    if not initial['physics_settings']['contacts'].get('robot_self_collisions'):
        return None  # Historical self-disabled trials cannot certify clearance.
    counts={'self':0,'inter_robot':0}
    for row in rows:
        report=row['observation'].get('robot_contacts')
        if not report or not report.get('reporting_enabled'):
            raise ValueError('Self-collision audit is missing robot-pair telemetry')
        for kind in counts:
            found=any(robot_pair(p['body0'],p['body1'])==kind and p['contact_samples']>0 for p in report['pairs'])
            if found!=report[kind+'_contact']: raise ValueError('Robot contact flags differ from body-pair records')
            counts[kind]+=int(found)
    return {'self_collisions_enabled':True,'steps':len(rows),'contact_steps':counts,
        'contact_free':not any(counts.values()),'scope':'Configured collision hulls; no positive clearance margin certified'}


def contact_costs(value=None):
    value={} if value is None else dict(value)
    if set(value)-{'self','inter_robot'}:raise ValueError('Unknown robot contact cost')
    result={key:float(value.get(key,0.)) for key in ('self','inter_robot')}
    if not np.isfinite(list(result.values())).all() or min(result.values())<0:
        raise ValueError('Robot contact costs must be finite and nonnegative')
    return result


def collision_cost(observation,costs):
    if not any(costs.values()):return 0.
    report=observation.get('robot_contacts')
    if not observation.get('physics_settings',{}).get('contacts',{}).get('robot_self_collisions') or not report or not report.get('reporting_enabled'):
        raise ValueError('Collision reward requires self-contact physics and robot-pair reports')
    return sum(costs[key]*bool(report[key+'_contact']) for key in costs)


def contact_terminal(observation,enabled,terminal,terminated,truncated,score):
    if not enabled:return terminal,terminated,truncated,score
    collision_cost(observation,{'self':1.,'inter_robot':1.})  # Require real contact reports.
    report=observation['robot_contacts']
    if not (report['self_contact'] or report['inter_robot_contact']):return terminal,terminated,truncated,score
    reason='inter_robot_contact' if report['inter_robot_contact'] else 'robot_self_contact'
    return -2.,True,False,{**score,'success':False,'complete_chess_move':False,'reason':reason}


class RobotContacts:
    def __init__(self):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.decode=PhysicsSchemaTools.intToSdfPath;self.clear()
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self.record)

    def clear(self): self.pairs={}

    def record(self,headers,data):
        for header in headers:
            a,b=sorted((str(self.decode(header.actor0)),str(self.decode(header.actor1))))
            kind=robot_pair(a,b)
            if kind is None or not header.num_contact_data: continue
            row=self.pairs.setdefault((a,b),{'body0':a,'body1':b,'kind':kind,'contact_samples':0,
                'sum_impulse_Ns':0.,'minimum_separation_m':0.,'collider_pairs':set(),
                'deepest_contact_position_m':None})
            colliders=tuple(sorted(str(self.decode(getattr(header,key,body))) for key,body in
                (('collider0',header.actor0),('collider1',header.actor1))))
            row['collider_pairs'].add(colliders)
            for contact in data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]:
                row['contact_samples']+=1
                row['sum_impulse_Ns']+=float(np.linalg.norm(contact.impulse))
                if float(contact.separation)<row['minimum_separation_m'] and hasattr(contact,'position'):
                    row['deepest_contact_position_m']=[float(v) for v in contact.position]
                row['minimum_separation_m']=min(row['minimum_separation_m'],float(contact.separation))

    def snapshot(self):
        return {'reporting_enabled':True,'scope':'last reset/restore or control step; robot-robot contacts only',
            'pairs':[{**self.pairs[key],'collider_pairs':[list(p) for p in sorted(self.pairs[key]['collider_pairs'])]}
                for key in sorted(self.pairs)],
            'self_contact':any(r['kind']=='self' for r in self.pairs.values()),
            'inter_robot_contact':any(r['kind']=='inter_robot' for r in self.pairs.values())}

    def close(self): self.subscription=None
