import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import yaml

default_colors=[(255,96,208),(1,0,255),(255,0,0),(255,255,0),(0,255,0),(160,128,96),(255,128,0),(153,0,153),(153,153,0),(102,0,0)] #default colors for plotting phases
default_colors_2=[(255,128,0),(153,0,153),(153,153,0),(102,0,0),(253,76,0),(0,25,51),(255,255,102)]


def create_new_data_frame(scorer,config_file):
    ''' creating a new data frame for data from 2 cameras'''
    with open(config_file) as f:
        cfg=yaml.full_load(f)
    body_part=cfg['upper_body_parts' ]
    for x in cfg['bottom_body_parts']:
        if x not in body_part:
            body_part.append(x)
    coordinate=['x','y','likelihood']
    col_levels=[scorer,body_part,coordinate]
    cols=pd.MultiIndex.from_product(col_levels)
    new_df=pd.DataFrame(index=range(100000),columns=cols)
    return new_df
    


def find_coords(frame,body_parts):
    '''finding coords of different body parts in a specific frame
    '''

    colors=[(255,96,208),(1,0,255),(255,0,0),(255,255,0),(0,255,0),(160,128,96),(255,128,0),(153,0,153),(153,153,0),(102,0,0)]

    loc_df=pd.DataFrame(index=['x','y'],columns=body_parts)
    loc_df.columns.name='bodyParts'
    
    for body_part,color in zip(body_parts,colors):
        all_ind=np.where(np.all(frame==color,axis=-1))
        all_x_ind=all_ind[0][:]
        all_y_ind=all_ind[1][:]
        x_ind=np.mean(all_x_ind)
        y_ind=np.mean(all_y_ind)
        loc_df[body_part]=np.around([y_ind,x_ind])
    return loc_df


def update_data_frame(frame_number,data_frame,update):
    ''' updating a data frame with new locations
    updade: data frame, columns=body parts, index='x','y'
    '''
    idx=pd.IndexSlice
    for col in update.columns:
        x,y=update.index[0],update.index[1]
        data_frame.loc[frame_number,idx[:,col,['x','y']]]=update[col][x],update[col][y] 
    
    

def fix_label(target_body_part,ref_body_part,target_frame_number,ref_frame_number,target_df,ref_df,new_data_frame,index):
    ''' fixing wrong labeling based on a corresponding frame from another camera, and updating the relevant data frame.
    
    inputs:
    target_body_part: the incorrectly labeled body part (string)
    ref_body_part: a reference body part for fixing (usually head/tail, from another camera) (string)
    target_frame_number: the frame number of the mislabeled bodypart (integer_)
    ref_frame_number: a corresponding frame from another camera (integer)
    target_df: data frame of target body_part (pandas data frame)
    ref_df: data frame with data from another camera
    
    '''
    #if likelihood is insufficient , fix the label coordinates    
    idx=pd.IndexSlice        
    if float(target_df.loc(axis=1)[:,target_body_part,'likelihood'].values[target_frame_number])<0.2: 
        ref_dist=ref_df.loc[ref_frame_number,idx[:,ref_body_part,['x','y']]].values-ref_df.loc[ref_frame_number,idx[:,target_body_part,['x','y']]].values           
        target_label=target_df.loc[target_frame_number,idx[:,ref_body_part,['x','y']]].values-ref_dist
        target_df.loc[target_frame_number,idx[:,target_body_part,['x','y']]]=target_label  #updating the data frame with the fixed label coords
        target_df.loc[target_frame_number,idx[:,target_body_part,['likelihood']]]='label was editted using fix_label' 
        
        update_likelihood(new_data_frame,index,target_body_part,0.5)
               

        
        
def merge_labels(target_frame,source_frame,target_frame_num,source_frame_num,target_df,source_df,config_file,new_data_frame,index,plot=True):
    '''
    merging frames from 2 different cameras, if plot=True the fnction plots the merged frame and the source frame 
    body_parts: the body parts of the source frame (can be imported from the config file)
    '''
    with open(config_file) as f:
        cfg=yaml.full_load(f)
    ref_point=cfg['ref_point']
    body_parts=cfg['bottom_body_parts']
    
    colors=[(255,128,0),(153,0,153),(153,153,0),(102,0,0),(253,76,0),(0,25,51),(255,255,102)]

    Idx=pd.IndexSlice
    source_ref_point=source_df.loc[source_frame_num,Idx[:,ref_point,['x','y']]].values
    target_ref_point=target_df.loc[target_frame_num,Idx[:,ref_point,['x','y']]].values
    
    source_dist={}
    target_loc=pd.DataFrame(index=['x','y'],columns=body_parts)
    target_loc.columns.name='bodyPart'
    for val in body_parts:
        source_dist[val]=source_df.loc[source_frame_num,Idx[:,val,['x','y']]].values-source_ref_point
        result=target_ref_point+source_dist[val]
        target_loc[val]=result[0],result[1]
        
    # plotting the estimated legs coordinates
    for body_part,color in zip(body_parts,colors):
        image=cv2.circle(target_frame,(int(target_loc[body_part]['x']),int(target_loc[body_part]['y'])),6,color,-1)
        
    if plot==True:
        f=plt.figure(figsize=(30,15))
        f.add_subplot(1,2,1)
        plt.title('merged frames')
        plt.imshow(target_frame)
        f.add_subplot(1,2,2)
        plt.imshow(source_frame)
        plt.title('source frame')
        
    for body_part in body_parts:
        if np.isnan(float(new_data_frame.loc[source_frame_num,Idx[:,body_part,'likelihood']])): ###index=source_frame_num vs target
            if body_part==cfg['ref_point']:
                update_likelihood(new_data_frame,index,body_part,1)
            else:
                update_likelihood(new_data_frame,index,body_part,0.75)
    
    all_body_parts=set(cfg['bottom_body_parts']+cfg['upper_body_parts'])########
    parts=[i for i in all_body_parts if i not in body_parts] ########       
    update_likelihood(new_data_frame,index,parts,1)
    
        
    return image,target_loc


        
def update_likelihood(data_frame,index,body_part,likelihood):
    Idx=pd.IndexSlice
    data_frame.loc[index,pd.IndexSlice[:,body_part,'likelihood']]=likelihood
    
        
        
    
    
    
        




