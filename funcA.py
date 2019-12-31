import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import yaml

default_colors=[(255,96,208),(1,0,255),(255,0,0),(255,255,0),(0,255,0),(160,128,96),(255,128,0),(153,0,153),(153,153,0),(102,0,0)] #default colors for plotting phases
default_colors_2=[(255,128,0),(153,0,153),(153,153,0),(102,0,0),(253,76,0),(0,25,51),(255,255,102)]

def init_project(project_path,bottom_data_path,upper_data_path,bottom_video_path,upper_video_path):

    cfg_data={'bottom_body_parts':['body_part1','body_part2'],
          'upper_body_parts':['body_part1','body_part2'],
          'ref_point': [],
          'dist_markers':  ['body_part_1', 'body_part_2'],
          'upper_first_frame':[],
          'bottom_first_frame' : [] }
    
    #with open (rf'{project_path}/config.yaml','w') as file:
    #   document=yaml.dump(cfg_data,file)
    print('a configuration file was created. you can find it in the path:',rf'{project_path}/config.yaml')


    df_bottom=pd.read_hdf(bottom_data_path) #creating data frame for bottom video data
    df_up=pd.read_hdf(upper_data_path)  #creating data frame for upper video datacap_b=cv2.VideoCapture(bottom_video_path)
    cap_b=cv2.VideoCapture(bottom_video_path)
    if not cap_b.isOpened():
            print('bottom cap can\\t open')
    total_frames_b=cap_b.get(7)
    print(f'total frames in the bottom video : {total_frames_b}')
    
    cap_u=cv2.VideoCapture(upper_video_path)
    if not cap_u.isOpened():
        print('upper video van\\t open')
    total_frames_u=cap_u.get(7)
    print('total frames in the upper video:'+str(total_frames_u))

    return df_bottom,df_up,cap_b,cap_u
    

def find_ratio(data_frames,config_file):
    '''finding the ratio between two frames
        inputs: data frames:[bottom_data_frame,upper_data_frame]
                config file: path of corresponding configuration file
    '''
    with open(config_file) as f:
        cfg=yaml.full_load(f)
    b_first_frame,u_first_frame=cfg['bottom_first_frame'],cfg['upper_first_frame']
    df_bottom,df_up=data_frames[0],data_frames[1]
    
    b_nose_x=df_bottom.loc(axis=1)[:,cfg['dist_markers'][0],'x'].values[:]
    b_nose_y=df_bottom.loc(axis=1)[:,cfg['dist_markers'][0],'y'].values[:]
    b_tail_x=df_bottom.loc(axis=1)[:,cfg['dist_markers'][1],'x'].values[:]
    b_tail_y=df_bottom.loc(axis=1)[:,cfg['dist_markers'][1],'y'].values[:]
    u_nose_x=df_up.loc(axis=1)[:,cfg['dist_markers'][0],'x'].values[:]
    u_nose_y=df_up.loc(axis=1)[:,cfg['dist_markers'][0],'y'].values[:]
    u_tail_x=df_up.loc(axis=1)[:,cfg['dist_markers'][1],'x'].values[:]
    u_tail_y=df_up.loc(axis=1)[:,cfg['dist_markers'][1],'y'].values[:]
    
    square_dist_b=(b_nose_x-b_tail_x)**2+(b_nose_y-b_tail_y)**2
    square_dist_u=(u_nose_x-u_tail_x)**2+(u_nose_y-u_tail_y)**2
    dist_u=np.sqrt(square_dist_u)
    dist_b=np.sqrt(square_dist_b)
    dist_u,dist_b=np.median(dist_u),np.median(dist_b)
    dist_ratio=dist_u/dist_b
    print('the ratio is: ' + str(dist_ratio))
    return dist_ratio

def plot_coords(current_frame,data_frame,body_parts,image,plot=False):
    '''plotting the coordinates on body parts
    '''

    colors=[(255,96,208),(1,0,255),(255,0,0),(255,255,0),(0,255,0),(160,128,96),(255,128,0),(153,0,153),(153,153,0),(102,0,0)]

    coords={}
    for body_part,color in zip(body_parts,colors):
        current_coord=np.around(data_frame.loc(axis=1)[:,body_part,['x','y']].values[int(current_frame)])
        if coords and any(np.all(np.absolute(np.array(list(coords.values()))-current_coord)<=3,axis=1)):
            current_coord=current_coord+3
            print(f'the {body_part} body part in {position} frame number {current_frame} changed its location due to an overlap')
        coords[body_part]=current_coord
        image=cv2.circle(image,(int(coords[body_part][0]),int(coords[body_part][1])),5,color,-1)

    if plot==True:
        f=plt.figure(figsize=(30,15))
        plt.imshow(image)
    return pd.DataFrame.from_dict(coords,orient='index',columns=['x','y']).T,image


def rotate_frame(input_mat,alpha,beta,gamma,dx,dy,dz,f,plot=False):
    ''' rotating frames '''
    
    alpha = (alpha - 90)*(np.pi/180)
    beta = (beta- 90)*(np.pi/180)
    gamma = (gamma- 90)*(np.pi/180)

    ## get width and height 
    h, w = input_mat.shape[:2]

    ## Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w/2],
             [0,1,-h/2],
             [ 0, 0,  0],
             [0, 0, 1 ]])
            
    ## Rotation matrices around the X, Y, and Z axis

    RX=np.array([[1,          0,           0, 0],
             [0, np.cos(alpha), -np.sin(alpha), 0],
             [0, np.sin(alpha),  np.cos(alpha), 0],
             [0,          0,           0, 1]
            ])
    RY=np.array([[np.cos(beta), 0, -np.sin(beta), 0],
             [0, 1,          0, 0],
             [np.sin(beta), 0,  np.cos(beta), 0],
             [ 0, 0,          0, 1]])
    
    RZ=np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
             [np.sin(gamma),  np.cos(gamma), 0, 0],
             [0,          0,           1, 0],
             [0,          0,           0, 1]])
    
    ## Composed rotation matrix with (RX, RY, RZ)
    R1 = (RX.dot(RY)).dot(RZ)

    ## Translation matrix
    T=np.array([[1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0,  1]])
    
    ## 3D -> 2D matrix
    A2=np.array([[f, 0, w/2, 0],
             [0, f, h/2, 0],
             [0, 0,   1, 0]])

    ## Final transformation matrix
    trans_mat = A2.dot(T.dot(R1.dot(A1)))

    # Apply matrix transformation
    output_mat=cv2.warpPerspective(input_mat, trans_mat, (w,h))
    
    if plot==True:
        f=plt.figure(figsize=(30,15))
        f.add_subplot(1,2,1)
        plt.imshow(output_mat)
        plt.title('rotated image')
        
        f.add_subplot(1,2,2)
        plt.imshow(input_mat)
        plt.title('original image')
    
    return output_mat




####################################################################################################################








