import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from funcA import init_project,find_ratio,plot_coords,rotate_frame
from funcB import find_coords, update_data_frame,fix_label, merge_labels,create_new_data_frame
import yaml

project_path='/home/nechama11/glass_project'
upper_data=project_path+'/u22.h5'
bottom_data=project_path+'/b22.h5'
upper_video=project_path+'/up22.MP4'
bottom_video=project_path+'/bottom22.MP4'

dfb,dfu,cap_b,cap_u=init_project(project_path,bottom_data,upper_data, bottom_video, upper_video)

config_file='/home/nechama11/glass_project/config.yaml'

scorer=['DeepCut_resnet50_2cameras_upSep16shuffle1_700000']
new_df=create_new_data_frame(scorer,config_file)



with open(config_file) as f:
    cfg=yaml.full_load(f)
    
    
#choosing frames
b=8899#the frame from the bottom video

cap_b.set(1,b)
ret, frameB = cap_b.read()
dist_ratio=find_ratio([dfb,dfu],config_file)
u=2*b+15 #the frame from the upper video
cap_u.set(1,u)
ret, frameU=cap_u.read()

u_loc, frameU=plot_coords(u,dfu,cfg['upper_body_parts'],frameU)
b_loc,frameB=plot_coords(b,dfb,cfg['bottom_body_parts'],frameB)



frameB =cv2.resize(frameB,None,fx=dist_ratio,fy=dist_ratio)

frameB=rotate_frame(frameB,90-180,90,90-4,0,0,24,24)
frameU=rotate_frame(frameU,90,90,90,0,0,45,45)

b_loc=find_coords(frameB,cfg['bottom_body_parts'])

update_data_frame(b,dfb,b_loc)

fix_label('TailBase','Head1',u,b,dfu,dfb,new_df,b)

image,target_loc=merge_labels(frameU,frameB,u,b,dfu,dfb,config_file,new_df,b,plot=True)

update_data_frame(b,new_df,target_loc)
update_data_frame(b,new_df,u_loc)


