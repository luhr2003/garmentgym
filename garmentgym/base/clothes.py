import os
from pathlib import Path
import random
import subprocess
import sys
import imageio

import trimesh
from garmentgym.base.config import Config
from garmentgym.utils.flex_utils import center_object
import pyflex
import numpy as np
from Imath import PixelType
from garmentgym.base.clothes_mesh import clothes_mesh
from garmentgym.utils.basic_utils import *

class Clothes:
    def __init__(self,name:str,mesh_category_path:str,config:Config,scale:int=1.2,need_urs:bool=False,gui:bool=True,random_choose:bool=True,domain_randomlization:bool=False,id:int=-1) -> None:
        self.scale=scale
        self.need_urs=need_urs
        self.gui=gui
        self.mesh_category_path=mesh_category_path
        self.random_choose=random_choose
        self.domain_randomlization=domain_randomlization
        self.id=id
        self.path=None
        self.name=name
        if id != -1:
            self.random_choose=False
        self.mesh=self.get_mesh(self.mesh_category_path,self.random_choose)

        if self.domain_randomlization:
            config.cloth_config.update({'cloth_mass':np.random.uniform(30,70),'cloth_stiff':np.random.uniform(0.2, 2.0)})
        self.mesh.set_config(config.cloth_config.cloth_pos,config.cloth_config.cloth_size_scale,config.cloth_config.cloth_mass,config.cloth_config.cloth_stiff)

        self.current_mesh=None
        self.flattened_area=-1
        self.current_height=-1
        self.current_width=-1
        self.init_position=None
        self.pre_cross_position=None
        self.bottom_right=-1
        self.bottom_left=-1
        self.top_right=-1
        self.top_left=-1
        self.right_shoulder=-1
        self.left_shoulder=-1
        self.keypoint=None


    def get_mesh(self,mesh_category_path:str,random_choose:bool):
        assert mesh_category_path is not None
        if random_choose:
            self.path = str(random.choice(list(Path(mesh_category_path).rglob('*processed.obj'))))
            self.id=int(self.path.split('/')[-2])
        else:
            self.path = os.path.join(mesh_category_path,str(self.id))
            self.path=str(list(Path(self.path).rglob('*processed.obj'))[0])
        
        return clothes_mesh(path=self.path,name=self.name,need_urs=self.need_urs)
    def flatten_cloth(self):
        positions = pyflex.get_positions().reshape(-1, 4)
        positions[:self.mesh.num_particles, :3] = self.mesh.vertices
        positions[:, 1] += 0.1
        pyflex.set_positions(positions)
        for _ in range(40):
            pyflex.step()
            if self.gui:
                pyflex.render()
        center_object()
    def init_info(self):
        self.init_position=pyflex.get_positions().reshape(-1,4)
        self.init_position=np.array(self.init_position).astype(np.float32)
        self.init_position[:,:3]=self.init_position[:,:3]@get_rotation_matrix(np.array([0,1,0]),np.pi/2)
        pyflex.set_positions(self.init_position.flatten())
        xzy=self.init_position.reshape(-1, 4)[:self.mesh.num_particles, :3]
        x = xzy[:, 0]
        y = xzy[:, 2]
        self.flattened_area=self.mesh.cloth_trimesh.area/2
        self.mesh.cloth_height=float(np.max(y) - np.min(y))
        self.mesh.cloth_width=float(float(np.max(x) - np.min(x)))
        self.get_keypoint_groups(xzy)
    
    def get_keypoint_groups(self,xzy : np.ndarray):
        x = xzy[:, 0]
        y = xzy[:, 2]

        cloth_height = float(np.max(y) - np.min(y))
        cloth_width = float(np.max(x) - np.min(x))
        
        max_ys, min_ys = [], []
        num_bins = 40
        x_min, x_max = np.min(x),  np.max(x)
        mid = (x_min + x_max)/2
        lin = np.linspace(mid, x_max, num=num_bins)
        for xleft, xright in zip(lin[:-1], lin[1:]):
            max_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].min())
            min_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].max())

        #plot the rate of change of the shirt height wrt x
        diff = np.array(max_ys) - np.array(min_ys)
        roc = diff[1:] - diff[:-1]

        #pad beginning and end
        begin_offset = num_bins//5
        end_offset = num_bins//10
        roc[:begin_offset] = np.max(roc[:begin_offset])
        roc[-end_offset:] = np.max(roc[-end_offset:])
        
        #find where the rate of change in height dips, it corresponds to the x coordinate of the right shoulder
        right_x = (x_max - mid) * (np.argmin(roc)/num_bins) + mid

        #find where the two shoulders are and their respective indices
        xzy_copy = xzy.copy()
        xzy_copy[np.where(np.abs(xzy[:, 0] - right_x) > 0.01), 2] = 10
        right_pickpoint_shoulder = np.argmin(xzy_copy[:, 2])
        right_pickpoint_shoulder_pos = xzy[right_pickpoint_shoulder, :]

        left_shoulder_query = np.array([-right_pickpoint_shoulder_pos[0], right_pickpoint_shoulder_pos[1], right_pickpoint_shoulder_pos[2]])
        left_pickpoint_shoulder = (np.linalg.norm(xzy - left_shoulder_query, axis=1)).argmin()
        left_pickpoint_shoulder_pos = xzy[left_pickpoint_shoulder, :]

        #top left and right points are easy to find
        pickpoint_top_right = np.argmax(x - y)
        pickpoint_top_left = np.argmax(-x - y)

        #to find the bottom right and bottom left points, we need to first make sure that these points are
        #near the bottom of the cloth
        pickpoint_bottom = np.argmax(y)
        diff = xzy[pickpoint_bottom, 2] - xzy[:, 2]
        idx = diff < 0.1
        locations = np.where(diff < 0.1)
        points_near_bottom = xzy[idx, :]
        x_bot = points_near_bottom[:, 0]
        y_bot = points_near_bottom[:, 2]

        #after filtering out far points, we can find the argmax as usual
        pickpoint_bottom_right = locations[0][np.argmax(x_bot + y_bot)]
        pickpoint_bottom_left = locations[0][np.argmax(-x_bot + y_bot)]

        self.bottom_right=pickpoint_bottom_right,
        self.bottom_left=pickpoint_bottom_left,
        self.top_right=pickpoint_top_right,
        self.top_left=pickpoint_top_left,
        self.right_shoulder=right_pickpoint_shoulder,
        self.left_shoulder=left_pickpoint_shoulder,
        

        # get middle point
        middle_point_pos=np.array([0,0.1,0])
        self.middle_point=self.find_nearest_index(middle_point_pos)

        # get left and right points
        middle_band=np.where(np.abs(self.init_position[:,2]-middle_point_pos[2])<0.1)
        self.left_x=np.min(self.init_position[middle_band,0])
        self.right_x=np.max(self.init_position[middle_band,0])
        self.left_point=self.find_nearest_index([self.left_x,0,0])
        self.right_point=self.find_nearest_index([self.right_x,0,0])

        # get top and bottom points
        x_middle_band=np.where(np.abs(self.init_position[:,0]-self.init_position[self.middle_point,0])<0.1)
        self.top_y=np.min(self.init_position[x_middle_band,2])
        self.bottom_y=np.max(self.init_position[x_middle_band,2])
        self.top_point=self.find_nearest_index([0,0,self.top_y])
        self.bottom_point=self.find_nearest_index([0,0,self.bottom_y])
        # self.top_point=np.argmax(self.init_position[x_middle_band,2])
        # self.bottom_point=np.argmin(self.init_position[x_middle_band,2])

        self.keypoint=[self.bottom_left,self.bottom_right,self.top_left,self.top_right,self.left_shoulder,self.right_shoulder,self.middle_point,self.left_point,self.right_point,self.top_point,self.bottom_point]



    def find_nearest_index(self,point):
        point=np.array(point).astype(np.float32)
        dist=np.linalg.norm(self.init_position[:,:3]-point,axis=1)
        return np.argmin(dist)
    
    def update_info(self):
        curr_pos = pyflex.get_positions()
        xzy = curr_pos.reshape(-1, 4)[:self.mesh.num_particles, :3]
        x = xzy[:, 0]
        y = xzy[:, 2]

        self.current_height = float(np.max(y) - np.min(y))
        self.current_width = float(np.max(x) - np.min(x))


    def get_cloth_size(self):
        return self.cloth_height,self.cloth_width
    
    def get_cloth_config(self):
        return self.mesh.get_config_dict()
    
    def get_current_cloth_mesh(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        vertices = positions[:, :3]
        faces = pyflex.get_faces().reshape((-1, 3))
        self.current_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return self.current_mesh
    
    def blender_render_cloth(cloth_mesh, resolution):
        output_prefix = '/tmp/' + str(os.getpid())
        obj_path = output_prefix + '.obj'
        cloth_mesh.export(obj_path)
        commands = [
            'blender',
            'cloth.blend',
            '-noaudio',
            '-E', 'BLENDER_EEVEE',
            '--background',
            '--python',
            'render_rgbd.py',
            obj_path,
            output_prefix,
            str(resolution)]
        with open(devnull, 'w') as FNULL:
            while True:
                try:
                    # render images
                    subprocess.check_call(
                        commands,
                        stdout=FNULL)
                    break
                except Exception as e:
                    print(e)
        # get images
        output_dir = Path(output_prefix)
        color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
        color = color[:, :, :3]
        # depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
        redstr = depth.channel('R', PixelType(PixelType.FLOAT))
        depth = np.fromstring(redstr, dtype=np.float32)
        depth = depth.reshape(resolution, resolution)
        return color, depth
