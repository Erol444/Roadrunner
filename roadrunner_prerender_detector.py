import cv2
import numpy as np
import sys
import time
import open3d
import math
import json

depth_size = 1280*720*2 # 720p uint16
color_size = 1920*1080*3 # 10808p BGR


width = 1280
height = 720
road_width = int(width / 5 * 3)


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



def Project(points, intrinsic, distortion):
    result = []
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    if len(points) > 0:
        result, _ = cv2.projectPoints(points, rvec, tvec, intrinsic, distortion)
    return np.squeeze(result, axis=1)



try:
    from depthai_library.depthai_helpers.projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")

pcl_converter = PointCloudVisualizer(intrinsics_720p, 1280, 720)


def point_to_plane_dist(x1, y1, z1, a, b, c, d, e):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    print(d.shape)
    return d/e
#right_rectified = cv2.flip(right_rectified, 1)

add_once = 0
box = None






def runVideo(fps, depth_path, video_path, metadata_path, prerender_path):
    with open(depth_path, 'rb') as depth_file, open(metadata_path, 'r') as metadata_file, open(prerender_path, 'rb') as prerender_file:
        video = cv2.VideoCapture(video_path)
        metadata = json.load(metadata_file)
        metadata_index = 0
        while True:
            
            t1 = time.time()

            # Read (all data of current event) depth and color frames
            depth_bytes = depth_file.read(depth_size)
            ret, color_frame = video.read()
            prerender_bytes = prerender_file.read(depth_size // 2)

            # End of streams
            if len(depth_bytes) < depth_size or ret == False or (len(prerender_bytes) < (depth_size // 2)) :
                print('End of stream...')
                break

            # Convert to appropriate np array
            depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape( (720, 1280) )
            #color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape( (1080, 1920, 3) )
            prerender_frame = np.frombuffer(prerender_bytes, dtype=np.uint8).reshape( (720, 1280) )

            # Preprocess depth to grayscale
            depth_grayscale = (65535 // depth_frame).astype(np.uint8)
            depth_grayscale = cv2.applyColorMap(depth_grayscale, cv2.COLORMAP_HOT)

            # Display both frames
            cv2.imshow('Depth (grayscale)', depth_grayscale)
            cv2.imshow('Color', color_frame)

            color_frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            color_frame_rgb = cv2.resize(color_frame_rgb, (1280, 720))
            #print(color_frame_rgb)

            
            # prerender frame
            cv2.imshow("Backprojection: ", prerender_frame)
            
            # dilate and erode
            kernel = np.ones((5,5), np.uint8) 
            prerender_frame = cv2.dilate(prerender_frame, kernel, iterations=5) 
            prerender_frame = cv2.erode(prerender_frame, kernel, iterations=3) 

            #

            total_num_pixels = road_width * height
            num_pixels_counted = 0
            #for y in range( 0, height ) :
            #    for x in range( (width - road_width) // 2, (width - road_width) // 2 + road_width ) :
            #        if prerender_frame[y, x] > 128:
            #            num_pixels_counted += 1


            #prerender_frame = cv2.bitwise_not(prerender_frame)

            # Connected components with stats.
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(prerender_frame, connectivity=4)

            # Find the largest non background component.
            # Note: range() starts from 1 since 0 is the background label.
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

            print('max size: ',  max_size, ' max label: ', max_label)
            if max_size > 20000 : 
                # get color image from that blob
                w = np.sqrt(max_size) 
                
                cx = int(centroids[max_label][0])
                cy = int(centroids[max_label][1])

                tl = ( int(cx - w // 2), int(cy - w // 2))
                br = ( int(cx + w // 2), int(cy + w // 2))

                print('tl: ', tl, ' br: ', br)

                pothole = color_frame_rgb[ tl[1]:br[1], tl[0]:br[0] ]
                
                if pothole.shape[0] > 0 and pothole.shape[1] > 0: 
                    cv2.imshow('Pothole', pothole)

                # print GPS data
                print(metadata["metadata"][metadata_index])

                cv2.waitKey(0)    
            #    nb_components[max_label]



            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 254
            params.maxThreshold = 255

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 500
            params.maxArea = 999999


            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1

            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.87

            # Filter by Inertia
            params.filterByInertia = False
            params.minInertiaRatio = 0.01

            # Create detector with params 
            detector = cv2.SimpleBlobDetector_create()

            # Detect blobs.
            keypoints = detector.detect(prerender_frame)

            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            prerendered_with_keypoints = cv2.drawKeypoints(prerender_frame, keypoints, np.array([]), (255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Show keypoints
            cv2.imshow("Keypoints", prerendered_with_keypoints)
            #cv2.waitKey(0)


            #cv2.imshow("Image", im)


            # Calculate final score (sum all keypoint areas)
            max_area = height * road_width / 10
            final_area = 0.0
            for keypoint in keypoints :
                approx_area = ((keypoint.size / 2) ** 2) * np.pi
                final_area += approx_area

            if final_area > 0 :
                norm_score =  final_area / max_area
            else :
                norm_score = 0
            if norm_score > 1 : 
                norm_score = 1

            good_score = 1 - norm_score

            #print('Max area: ', max_area, ' final area: ', final_area, ' norm_score: ', norm_score)
            print('Score: ', good_score)

            print('Alternate score: ', 1 - (num_pixels_counted / total_num_pixels))


            if good_score < 0.8 :
                cv2.waitKey(0)



            metadata_index += 1

            to_wait = int( ( (1.0 / fps) * 1000 - (time.time() - t1) * 1000) )
            if to_wait <= 0:
                to_wait = 1
            cv2.waitKey(to_wait +10)



fps = int(sys.argv[1])
runVideo(fps, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
cv2.waitKey(0)
