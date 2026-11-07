"""Batched opposing-finger midpoint for consistent state augmentation."""
import torch
from torch import nn


class OpeningMidpoint(nn.Module):
    def __init__(self,specification,anchors):
        super().__init__()
        self.joints=specification['joints']
        self.names=specification['names']
        self.root=specification['tool_link'].rsplit('/',1)[0]
        self.tool_link=specification['tool_link']
        self.register_buffer('anchors',torch.as_tensor(anchors,dtype=torch.float32))
        for index,joint in enumerate(self.joints):
            for name in ['f0','f1_inverse','parent_world']:
                self.register_buffer(f'{name}_{index}',torch.as_tensor(joint[name],dtype=torch.float32))
            x,y,z=joint['axis']
            self.register_buffer(f'cross_{index}',torch.tensor([[0.,-z,y],[z,0.,-x],[-y,x,0.]],dtype=torch.float32))

    def forward(self,q,jaw2):
        return self.forward_pose(q,jaw2)[0]

    def forward_pose(self,q,jaw2):
        batch=q.shape[0]
        frames={}
        for index,joint in enumerate(self.joints):
            angle=jaw2 if joint['name']=='jaw2_hinge' else q[:,self.names.index(joint['name'])]
            angle=torch.deg2rad(angle)[:,None,None]
            cross=getattr(self,f'cross_{index}')
            rotation=torch.eye(3,device=q.device)[None]+torch.sin(angle)*cross+(1-torch.cos(angle))*(cross@cross)
            matrix=torch.eye(4,device=q.device)[None].repeat(batch,1,1)
            matrix[:,:3,:3]=rotation.transpose(1,2)
            parent=frames.get(joint['parent'],getattr(self,f'parent_world_{index}'))
            frames[joint['child']]=getattr(self,f'f1_inverse_{index}')@matrix@getattr(self,f'f0_{index}')@parent
        points=[]
        for index in range(2):
            point=torch.cat([self.anchors[index],torch.ones(1,device=q.device)])
            points.append((point@frames[f'{self.root}/jaw_{index+1}'])[:,:3])
        return (points[0]+points[1])*.5,frames[self.tool_link][:,:3,:3]
