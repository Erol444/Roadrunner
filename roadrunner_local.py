import cv2
import numpy as np
import sys
import time



depth_size = 1280*720*2 # 720p uint16
color_size = 1920*1080*3 # 10808p BGR


def run(fps, depth_path, color_path):
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


fps = int(sys.argv[1])
run(fps, sys.argv[2], sys.argv[3])
cv2.waitKey(0)



