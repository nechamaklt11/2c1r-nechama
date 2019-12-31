project_path: /home/nechama11/glass_project 

upper_data_path: /home/nechama11/glass_project/u22.h5
bottom_data_path: /home/nechama11/glass_project/b22.h5
upper_video_path: /home/nechama11/glass_project/up22.MP4
bottom_video_path: /home/nechama11/glass_project/bottom22.MP4


upper_body_parts: #body parts appearing in the upper video
- Head1
- Head2
- Head3
- Back1
- Back2
- TailBase

bottom_body_parts: #body parts appearing in the bottom video
- Head1
- TailBase
- ForepawR
- ForepawL
- HindpawR
- HindpawL

ref_point: TailBase

dist_markers: [Head1,TailBase] #vector with two columns, represents the body parts that will be used for finding the ratio between the two cameras


upper_first_frame: 1100 #first frame with a rat

bottom_first_frame: 550 #first frame with a rat






