import cv2
import numpy as np
import sys
import time
import open3d

depth_size = 1280*720*2 # 720p uint16
color_size = 1920*1080*3 # 10808p BGR

# Camera intrinsics
intrinsics_720p = [
    [861.674072265625, 0.0, 638.2583618164062],
    [0.0, 862.1583862304688, 364.5867919921875],
    [0.0, 0.0, 1.0]
]

# Camera intrinsics
intrinsics_800p = [
    [861.674072, 0.0, 638.258362],
    [0.0, 862.158386, 404.586792],
    [0.0, 0.0, 1.0]
]


try:
    from depthai_helpers.projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")

pcl_converter = PointCloudVisualizer(intrinsics_720p, 1280, 720)

#right_rectified = cv2.flip(right_rectified, 1)



add_once
once = 0
box = None

def runVideo(fps, depth_path, video_path):
    with open(depth_path, 'rb') as depth_file:
        video = cv2.VideoCapture(video_path)
        while True:         
            # Read (all data of current event) depth and color frames
            depth_bytes = depth_file.read(depth_size)
            ret, color_frame = video.read()

            # End of streams
            if len(depth_bytes) < depth_size or ret == False : 
                print('End of stream...')
                break

            # Convert to appropriate np array
            depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape( (720, 1280) )
            #color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape( (1080, 1920, 3) )    

            # Preprocess depth to grayscale
            depth_grayscale = (65535 // depth_frame).astype(np.uint8)
            depth_grayscale = cv2.applyColorMap(depth_grayscale, cv2.COLORMAP_HOT)

            # Display both frames
            cv2.imshow('Depth (grayscale)', depth_grayscale)
            cv2.imshow('Color', color_frame)

            color_frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            color_frame_rgb = cv2.resize(color_frame_rgb, (1280, 720))
            #print(color_frame_rgb)

            # Processing
            # Visualize depth
            pcl_converter.rgbd_to_projection(depth_frame, color_frame_rgb)
            
            global add_once
            if add_once == 0 :
                box = open3d.geometry.TriangleMesh.create_box(1, 1, 0.01)
                pcl_converter.vis.add_geometry(box)
                add_once = 1
            else :
                center = box.get_center()
                #center[0] += 1
                box.translate([1,0,0])
                pcl_converter.vis.update_geometry(box)
                print('Translate, center: ', center)
                

            pcl_converter.visualize_pcd()
            

            




            cv2.waitKey( int((1.0 / fps) * 1000) )



fps = int(sys.argv[1])
runVideo(fps, sys.argv[2], sys.argv[3])
cv2.waitKey(0)













def runColor(fps, depth_path, color_path):
    with open(depth_path, 'rb') as depth_file, open(color_path, 'rb') as color_file:
        while True:     
            # Read (all data of current event) depth and color frames
            depth_bytes = depth_file.read(depth_size)
            color_bytes = color_file.read(color_size)
            print('File size: depth: ', len(depth_bytes), ' color: ', len(color_bytes))

            # End of streams
            if len(depth_bytes) < depth_size or len(color_bytes) < color_size or not depth_bytes:
                print('End of stream...')
                break

            # Convert to appropriate np array
            depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape( (720, 1280) )
            color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape( (1080, 1920, 3) )    

            # Preprocess depth to grayscale
            depth_grayscale = (65535 // depth_frame).astype(np.uint8)
            depth_grayscale = cv2.applyColorMap(depth_grayscale, cv2.COLORMAP_HOT)

            # Display both frames
            cv2.imshow('Depth (grayscale)', depth_grayscale)
            cv2.imshow('Color', color_frame)

            cv2.waitKey( int((1.0 / fps) * 1000) )

