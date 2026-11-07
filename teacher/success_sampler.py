"""SGS-style reset selection; outcomes come only from training episodes."""
from collections import deque
import copy
import numpy as np


class SuccessSampler:
    def __init__(self,names,window=100,mode=.5,concentration=1.,temperature=2.,uniform_mix=.2):
        if (not names or len(set(names))!=len(names) or window<1 or not 0<mode<1
                or not np.isfinite([concentration,temperature,uniform_mix]).all()
                or concentration<0 or temperature<1 or not 0<=uniform_mix<=1):
            raise ValueError('Invalid success sampler configuration')
        self.names=list(names);self.history={name:deque(maxlen=window) for name in names}
        self.window=window;self.mode=mode;self.concentration=concentration
        self.temperature=temperature;self.uniform_mix=uniform_mix
    def probabilities(self):
        # Unknown configurations start at the kernel's mode, so they get tried.
        rates=np.array([np.mean(self.history[n]) if self.history[n] else self.mode for n in self.names])
        weights=((rates+1e-4)**(self.concentration*self.mode)
                 *(1-rates+1e-4)**(self.concentration*(1-self.mode)))**(1/self.temperature)
        probabilities=weights/weights.sum()
        return (1-self.uniform_mix)*probabilities+self.uniform_mix/len(self.names)
    def sample(self,rng):
        return int(rng.choice(len(self.names),p=self.probabilities()))
    def record(self,name,success):
        if name not in self.history or type(success) not in (bool,np.bool_):
            raise ValueError('Only known training resets and Boolean outcomes are accepted')
        self.history[name].append(bool(success))
    def state(self):
        return {'window':self.window,'mode':self.mode,'concentration':self.concentration,
            'temperature':self.temperature,'uniform_mix':self.uniform_mix,
            'history':{n:list(v) for n,v in self.history.items()},
            'probabilities':dict(zip(self.names,self.probabilities().tolist()))}
