import argparse
import pickle
import random
import sys
import time
import os

import cv2
import tqdm

curpath=os.getcwd()
sys.path.append(curpath)

import open3d as o3d
from garmentgym.garmentgym.utils.init_env import init_env
import pyflex
from garmentgym.garmentgym.base.clothes_env import ClothesEnv
from garmentgym.garmentgym.base.clothes import Clothes
from copy import deepcopy
from garmentgym.garmentgym.clothes_hyper import hyper
from garmentgym.garmentgym.base.config import *
from garmentgym.garmentgym.utils.exceptions import MoveJointsException
from garmentgym.garmentgym.utils.flex_utils import center_object, wait_until_stable
from multiprocessing import Pool,Process
from garmentgym.garmentgym.utils.translate_utils import pixel_to_world, pixel_to_world_hard, world_to_pixel, world_to_pixel_hard
from garmentgym.garmentgym.utils.basic_utils import make_dir
task_config = {"task_config": {
    'observation_mode': 'cam_rgb',
    'action_mode': 'pickerpickplace',
    'num_picker': 1,
    'render': True,
    'headless': False,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

from garmentgym.garmentgym.base.record import task_info



class FoldEnv(ClothesEnv):
    def __init__(self,mesh_category_path:str,gui=True,store_path="./",id=-1):
        self.config=Config(task_config)
        self.id=id
        self.clothes=Clothes(name="cloth"+str(id),config=self.config,mesh_category_path=mesh_category_path,id=id)
        super().__init__(mesh_category_path=mesh_category_path,config=self.config,clothes=self.clothes)
        self.store_path=store_path
        self.empty_scene(self.config)
        self.gui=gui
        self.gui=self.config.basic_config.gui
        center_object()
        self.action_tool.reset([0,0.1,0])
        pyflex.step()
        if gui:
            pyflex.render()
        
        self.info=task_info()
        self.action=[]
        self.info.add(config=self.config,clothes=self.clothes)
        self.info.init()
        
        self.grasp_states=[True,True]
        
    def record_info(self):
        self.info.update(self.action)
        make_dir(os.path.join(self.store_path,str(self.id)))
        self.curr_store_path=os.path.join(self.store_path,str(self.id),str(len(self.action))+".pkl")
        with open(self.curr_store_path,"wb") as f:
            pickle.dump(self.info,f)
    
    def get_cur_info(self):
        self.info.update(self.action)
        return self.info

        
    
    def pick_and_place_primitive(
        self, p1, p2, lift_height=0.2):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1]=0.03
        place_pos[1]=0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([preplace_pos], speed=2e-2)
        self.movep([place_pos], speed=1e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=10e-2)
        self.hide_end_effectors()
        
    def top_pick_and_place_primitive(
        self, p1, p2, lift_height=0.3):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1] += 0.06
        place_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([preplace_pos], speed=2e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()
    
    
    def pick_and_change_route(
        self, p1, p2,p3 ,lift_height=0.3):
        # prepare primitive params
        pick_pos, mid_pos,place_pos = p1.copy(), p2.copy(),p3.copy()
        pick_pos[1] -= 0.04
        place_pos[1] += 0.03 + 0.05
        mid_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        premid_pos = mid_pos.copy()
        premid_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([premid_pos], speed=2e-2)
        self.movep([mid_pos], speed=2e-2)
        
        self.movep([premid_pos], speed=1e-2)
    
        self.movep([preplace_pos], speed=1e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()
    

    def pick_change_nodown(
        self, p1, p2,p3 ,lift_height=0.3):
        # prepare primitive params
        pick_pos, mid_pos,place_pos = p1.copy(), p2.copy(),p3.copy()
        pick_pos[1] -= 0.04
        place_pos[1] += 0.03 + 0.05
        mid_pos[1] += 0.03 + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        premid_pos = mid_pos.copy()
        premid_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.movep([prepick_pos], speed=8e-2)
        self.movep([pick_pos], speed=6e-2)
        self.set_grasp(True)
        self.movep([prepick_pos], speed=1e-2)
        self.movep([premid_pos], speed=2e-2)

        self.movep([preplace_pos], speed=1e-2)
        self.movep([place_pos], speed=2e-2)
        self.set_grasp(False)
        self.movep([preplace_pos], speed=8e-2)
        self.hide_end_effectors()



    
    
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    def two_hide_end_effectors(self):
        self.two_movep([[0.5, 0.5, -1],[0.5,0.5,-1]], speed=5e-2)

    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()
             
    
    def step_fn(gui=True):
        pyflex.step()
        if gui:
            pyflex.render()
    def show_position(self):
        self.action_tool.shape_move(np.array([0.9,0,0.9]))
    
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 2
        target_pos=pos
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            if self.gui:
                pyflex.render()
            action=np.array(action)
            self.action_tool.step(action)
        raise MoveJointsException
    
    def execute_action(self,action):
        function=action[0]
        arg=action[1]
        if function=="pick_and_place_primitive":
            self.pick_and_place_primitive(arg[0],arg[1])
        elif function=="top_pick_and_place_primitive":
            self.top_pick_and_place_primitive(arg[0],arg[1])
        elif function=="pick_and_change_route":
            self.pick_and_change_route(arg[0],arg[1],arg[2])
        elif function=="pick_change_nodown":
            self.pick_change_nodown(arg[0],arg[1],arg[2])
        
    
    
        
    

    
    
    
    