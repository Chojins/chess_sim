"""SAC with small bounded task-space exploration and unchanged neural mean."""
import torch
from stable_baselines3.sac.policies import Actor,SACPolicy


class BoundedActor(Actor):
    log_std_min=-5.
    log_std_max=-2.5
    def get_action_dist_params(self,obs):
        mean,log_std,kwargs=super().get_action_dist_params(obs)
        return mean,torch.clamp(log_std,self.log_std_min,self.log_std_max),kwargs


class BoundedSACPolicy(SACPolicy):
    def make_actor(self,features_extractor=None):
        kwargs=self._update_features_extractor(self.actor_kwargs,features_extractor)
        return BoundedActor(**kwargs).to(self.device)
