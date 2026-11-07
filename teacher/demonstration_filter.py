"""Exclude settled nonterminal waits from memoryless-policy BC/prior labels."""
import numpy as np


def settled_dwell(observations,actions,released):
    x=np.asarray(observations);a=np.asarray(actions)
    if x.ndim!=2 or x.shape[1]!=76 or a.shape!=(len(x),6): raise ValueError('Expected Cartesian observations/actions')
    home=np.asarray(released,dtype=bool)&(np.abs(x[:,62:67]*90).max(axis=1)<5)
    return ((np.abs(a[:,:5]).max(axis=1)<.02)&(np.abs(x[:,11]*60)<3)
            &(a[:,5]==x[:,34])&~home)
