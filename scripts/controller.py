#!/usr/bin/env python

import os
import sys
import torch
sys.path.append('/usr/lib/python2.7/dist-packages')
sys.path.append('/home/rohan/rohan_m15x/projects/mujoco')
import rospy
from ros_ml_controller.srv import *

class StepForward:
    def __init__(self, model):
        self.load(model)
        return

    def load(self, path_to_policy):
        print("Loading... (weights: {})".format(path_to_policy))
        self.policy = torch.load_state_dict(os.path.join(path_to_policy, "actor.pt"))
        print("Loading... Done.")
        return

    @torch.no_grad()
    def step(self, obs):
        raw = self.policy.forward(torch.Tensor(obs), deterministic=True)
        raw_arr = raw.detach().numpy()
        return raw_arr

def callback(req):
    action = ctrl.step(req.observation)
    resp = ControlStepResponse()
    resp.action = action.tolist()
    return resp

if __name__ == '__main__':

    if len(sys.argv)!=2:
        print("Usage: {} <path_to_trained_model>".format(sys.argv[0]))
        sys.exit(1)

    ctrl = StepForward(sys.argv[1])

    rospy.init_node('rl_server')
    s = rospy.Service('~step_nn', ControlStep, callback)
    rospy.spin()
    
