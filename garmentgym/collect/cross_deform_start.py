import argparse
import subprocess
import os

import tqdm 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mesh_path', type=str,default="/home/luhr/garmentgym/garmentgym/cloth3d/train")
    parser.add_argument('--store_path', type=str,default="./cloth3d_val_data")
    parser.add_argument('--prefix', type=str,default="")
    parser.add_argument('--iter',type=int,default=0)
    args=parser.parse_args()
    mesh_path=args.mesh_path
    store_path=args.store_path
    prefix=args.prefix
    iter=args.iter
    mesh_list=sorted(os.listdir(mesh_path))
    for i in tqdm.tqdm(mesh_list):
        try:
            command="/home/luhr/anaconda3/envs/softgym/bin/python /home/luhr/garmentgym/garmentgym/garmentgym/collect/cross_deform.py --mesh_path {} --store_path {} --prefix {} --iter {} --mesh_id {}".format(mesh_path,store_path,prefix,iter,i)
            command_list=command.split(" ")
            p=subprocess.Popen(command_list)
            p.wait(1500)
        except subprocess.TimeoutExpired:
            print("an process terminated")
            p.kill()
            continue