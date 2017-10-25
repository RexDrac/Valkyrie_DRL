# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:17:53 2017

@author: chuanyu
"""

import valkyrie_gym_env
from valkyrie_gym_env import Valkyrie
import numpy.random as rand
import numpy as np

env = Valkyrie()
a=rand.randn(7)
#a=np.array(7)
a=np.zeros((7,))
#env.calCOMPos()
env.getLinkMass()
env.getLinkCOMPos()
env.getLinkCOMVel()
print(env.calCOMPos())
print(env.calCOMVel())
print(np.array(env.getObservation())-np.array(env.getFilteredObservation()))
for i in range(2000):
    _,reward,_,_ = env._step(a)
    #print(reward)
#env.calCOMPos()
env.getLinkMass()
env.getLinkCOMPos()
env.getLinkCOMVel()
print(env.linkMass)
print(env.calCOMPos())
print(env.calCOMVel())
env.linkMass["pelvisBase"]
print(np.array(env.getObservation())-np.array(env.getFilteredObservation()))

