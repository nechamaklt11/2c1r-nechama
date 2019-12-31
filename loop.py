import cv2
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from func_h5 import init_project,find_ratio,plot_coords,rotate_frame,find_coords,merge_frames,merge_data_frames,create_new_data_frame

project_path='/home/nechama11/glass_project'
upper_data=project_path+'/u22.h5'
bottom_data=project_path+'/b22.h5'
upper_video=project_path+'/up22.MP4'
bottom_video=project_path+'/bottom22.MP4' 

dfb,dfu,cap_b,cap_u=init_project(project_path,bottom_data,upper_data, bottom_video, upper_video)

config_file='/home/nechama11/glass_project/config.yaml'
scorer=['DeepCut_resnet50_2cameras_upSep16shuffle1_700000']
df=create_new_data_frame(scorer,config_file)

with open(config_file) as f:
    cfg=yaml.full_load(f)
    
dist_ratio=find_ratio([dfb,dfu],config_file)

for curr_frame_b in range(cfg['bottom_first_frame']+20,cfg['bottom_first_frame']+30):  #cap_b.get(7)
    cap_b.set(1,curr_frame_b)
    ret, frameB = cap_b.read()
    
    curr_frame_u=2*curr_frame_b+8 ############use a variable!!!!
    cap_u.set(1,curr_frame_u)
    ret, frameU = cap_u.read()
        
    u_loc, frameU=plot_coords(u,dfu,cfg['upper_body_parts'],frameU)
    b_loc,frameB=plot_coords(b,dfb,cfg['bottom_body_parts'],frameB)
    
    frameB =cv2.resize(frameB,None,fx=dist_ratio,fy=dist_ratio)
    
    frameB=rotate_frame(frameB,90-180,90,90-4,0,0,24,24)
    b_loc=find_coords(frameB,cfg['bottom_body_parts'])
    update_data_frame(b,dfb,b_loc)
    fix_label('TailBase','Head1',u,b,dfu,dfb,new_df,b)
    
    
    image,target_loc=merge_labels(frameU,frameB,u,b,dfu,dfb,config_file,new_df,b,plot=True)    df=merge_data_frames(curr_frame_b,loc_df,u_loc,df,scorer,config_file)
    update_data_frame(b,new_df,target_loc)
    update_data_frame(b,new_df,u_loc)
    
    if curr_frame_b%150==0:
        print(f'bottom frame={curr_frame_b}')
    
    
    
