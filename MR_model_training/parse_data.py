import glob
import numpy as np
import trimesh
files=glob.glob('data/*.txt')
for file in files:
    print(file)
    with open(file, "r") as f:
        lines=f.readlines()
    lines_np=[]
    for line in lines[:100]:
        line=line.replace(')(', ',')
        line=line.replace('(', '')
        line=line.replace(')', '')
        line_np=np.array(list(map(float, line.split(','))))
        lines_np.append(line_np)
    hand_joints=np.stack(lines_np)
    hand_joints=hand_joints.reshape(100, 78//3, 3)
    print(hand_joints.shape)
    for frameid, perframe_pts in enumerate(hand_joints):
        perframe_pts_tri = trimesh.PointCloud(vertices=perframe_pts)
        perframe_pts_tri.export(f'tmp/{frameid:04d}.ply')
    exit(0)
# file=''