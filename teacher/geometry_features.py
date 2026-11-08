"""Compact continuous geometry features; no gates, phases or action overrides."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def geometry_values(x):
    # Keep continuous task geometry and actual jaw feedback. Cartesian motion
    # no longer has to infer a pitch sum from unrelated individual joint angles.
    pitch=x[:,1:4].sum(1,keepdim=True)*2
    roll=x[:,4:5]*2
    jaw=x[:,5:6]*18
    commanded_jaw=jaw+x[:,28:29]*.5
    return torch.cat([jaw,x[:,11:15],x[:,17:23],commanded_jaw,x[:,34:35],
        x[:,68:76],pitch,roll,x[:,42:58]],dim=1)


class GeometryFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,mean,scale):
        super().__init__(observation_space,features_dim=len(mean))
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        if self.mean.shape!=(39,) or self.scale.shape!=self.mean.shape or torch.any(self.scale<=0):
            raise ValueError('Invalid compact geometry normalization')
    def forward(self,observations):
        return ((geometry_values(observations)-self.mean)/self.scale).clamp(-20,20)


def relational_values(x):
    """Continuous geometric distances, with no thresholds or action rules."""
    lateral=torch.linalg.vector_norm(x[:,12:14]*.03,dim=1,keepdim=True)/.01
    remaining=torch.linalg.vector_norm(x[:,18:20]*.1,dim=1,keepdim=True)/.05
    tool_height=(x[:,17:18]*.03+.005*x[:,70:71]-x[:,14:15]*.03)/.1
    return torch.cat([geometry_values(x),lateral,remaining,tool_height],dim=1)


class RelationalGeometryFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,mean,scale):
        super().__init__(observation_space,features_dim=42)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        if self.mean.shape!=(42,) or self.scale.shape!=(42,) or torch.any(self.scale<=0):
            raise ValueError('Invalid relational geometry normalization')
    def forward(self,observations):
        return ((relational_values(observations)-self.mean)/self.scale).clamp(-20,20)


def posture_values(x):
    # Arm configuration and commanded/home joint errors disambiguate paths
    # with the same tool position. No collision response or action rule here.
    return torch.cat([relational_values(x),x[:,:5]*2,x[:,23:28],x[:,62:67]],dim=1)


class PostureGeometryFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,mean,scale):
        super().__init__(observation_space,features_dim=57)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        if self.mean.shape!=(57,) or self.scale.shape!=(57,) or torch.any(self.scale<=0):
            raise ValueError('Invalid posture geometry normalization')
    def forward(self,observations):
        return ((posture_values(observations)-self.mean)/self.scale).clamp(-20,20)
