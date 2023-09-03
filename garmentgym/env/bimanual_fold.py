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
    'num_picker': 2,
    'render': True,
    'headless': False,
    'horizon': 100,
    'action_repeat': 8,
    'render_mode': 'cloth',
}}

from garmentgym.garmentgym.base.record import task_info



class BimanualFoldEnv(ClothesEnv):
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




    def two_pick_change_nodown(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.3):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=6e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=2e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=8e-2)
        self.two_hide_end_effectors()







    def two_pick_and_place_primitive(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.3,down_height=0.05):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += down_height
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=3e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=4e-3)  # 修改此处
        self.two_movep([preplace_pos1, preplace_pos2], speed=4e-3)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=4e-3)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    
    
    def two_pick_and_down(self, p1_s,p1_m ,p1_e, p2_s,p2_m,p2_e,lift_height=0.5,down_height=0.03):
    # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1_s.copy(),p1_m.copy(), p1_e.copy()
        pick_pos2, mid_pos2,place_pos2 = p2_s.copy(),p2_m.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        mid_pos1[1]+=down_height+0.04
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += 0.03
        mid_pos2[1]+=0.03
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1]=lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=6e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        self.two_movep([mid_pos1,mid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    
    
    
    
    
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
    
    def two_pick_change_nodown(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.3):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=6e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=2e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=8e-2)
        self.two_hide_end_effectors()

    def two_nodown_one_by_one(
        self, p1, p2,p3 ,p4,p5,p6,lift_height=0.3):
        # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1.copy(), p2.copy(),p3.copy()
        pick_pos2, mid_pos2,place_pos2 = p4.copy(), p5.copy(),p6.copy()
        pick_pos1[1] -= 0.04
        place_pos1[1] += 0.03 + 0.05
        mid_pos1[1] += 0.03 + 0.05
        pick_pos2[1] -= 0.04
        place_pos2[1] += 0.03 + 0.05
        mid_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp(False)
        self.two_movep([prepick_pos1,prepick_pos2], speed=8e-2)
        self.two_movep([pick_pos1,pick_pos2], speed=1e-2)
        self.set_grasp(True)
        self.two_movep([prepick_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([premid_pos1,prepick_pos2], speed=1e-2)
        self.two_movep([preplace_pos1,premid_pos2], speed=1e-2)
        self.two_movep([place_pos1,preplace_pos2], speed=1e-2)
        self.two_movep([place_pos1,place_pos2], speed=1e-2)
        self.set_grasp(False)
        self.two_movep([preplace_pos1,preplace_pos2], speed=1e-2)
        self.two_hide_end_effectors()
        
    def two_one_by_one(self, p1_s, p1_e, p2_s,p2_e,lift_height=0.3,down_height=0.03):
    # prepare primitive params
        pick_pos1, place_pos1 = p1_s.copy(), p1_e.copy()
        pick_pos2, place_pos2 = p2_s.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += down_height
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=10e-1)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=4e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([preplace_pos1,prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1,prepick_pos2], speed=1e-2) 
        self.set_grasp([False,True])
        self.two_movep([prepick_pos1,preplace_pos2], speed=1e-2) 
        self.two_movep([prepick_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, preplace_pos2], speed=5e-1)  # 修改此处
        self.two_hide_end_effectors()






    
    
    
    def two_pick_and_down(self, p1_s,p1_m ,p1_e, p2_s,p2_m,p2_e,lift_height=0.5,down_height=0.03):
    # prepare primitive params
        pick_pos1, mid_pos1,place_pos1 = p1_s.copy(),p1_m.copy(), p1_e.copy()
        pick_pos2, mid_pos2,place_pos2 = p2_s.copy(),p2_m.copy(), p2_e.copy()

        pick_pos1[1] += down_height
        mid_pos1[1]+=down_height+0.04
        place_pos1[1] += 0.03 + 0.05
        pick_pos2[1] += 0.03
        mid_pos2[1]+=0.03
        place_pos2[1] += 0.03 + 0.05

        prepick_pos1 = pick_pos1.copy()
        prepick_pos1[1] = lift_height
        premid_pos1 = mid_pos1.copy()
        premid_pos1[1] = lift_height
        preplace_pos1 = place_pos1.copy()
        preplace_pos1[1] = lift_height
        prepick_pos2 = pick_pos2.copy()
        prepick_pos2[1] = lift_height
        premid_pos2 = mid_pos2.copy()
        premid_pos2[1]=lift_height
        preplace_pos2 = place_pos2.copy()
        preplace_pos2[1] = lift_height

        # execute action
        self.set_grasp([False, False])
        self.two_movep([prepick_pos1, prepick_pos2], speed=8e-2)  # 修改此处
        self.two_movep([pick_pos1, pick_pos2], speed=6e-2)  # 修改此处
        self.set_grasp([True, True])
        self.two_movep([prepick_pos1, prepick_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        self.two_movep([mid_pos1,mid_pos2], speed=1e-2)  # 修改此处
        self.two_movep([premid_pos1,premid_pos2], speed=2e-2)  # 修改此处
        
        self.two_movep([preplace_pos1, preplace_pos2], speed=1e-2)  # 修改此处
        self.two_movep([place_pos1, place_pos2], speed=1e-2)  # 修改此处
        self.set_grasp([False, False])
        self.two_movep([preplace_pos1, preplace_pos2], speed=8e-2)  # 修改此处
        self.two_hide_end_effectors()
    
    def two_movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.08
        target_pos = np.array(pos)
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
            action = np.array(action)
            self.action_tool.step(action)


        raise MoveJointsException
    
    
    
    
    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1]], speed=5e-2)
        
    def two_hide_end_effectors(self):
        self.set_colors([False,False])
        self.two_movep([[0.5, 0.5, -1],[0.5,0.5,-1]], speed=5e-2)
    
    def execute_action(self,action):
        function=action[0]
        args=action[1]
        if function=="two_pick_and_place_primitive":
            self.two_pick_and_place_primitive(*args)
        elif function=="two_pick_and_down":
            self.two_pick_and_down(*args)
        elif function=="two_one_by_one":
            self.two_one_by_one(*args)
        elif function=="two_nodown_one_by_one":
            self.two_nodown_one_by_one(*args)
        elif function=="two_pick_change_nodown":
            self.two_pick_change_nodown(*args)
        
        
        
        
    
    
        
    

    
    
    
    