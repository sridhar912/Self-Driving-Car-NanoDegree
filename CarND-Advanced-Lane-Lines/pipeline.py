from CameraCalibration import cameraCalib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PrespectiveTransform import prespectiveTransform
from LaneInformation import laneInfo
from LaneDetection import LaneDetection
from glob import glob
from moviepy.editor import VideoFileClip

camera = cameraCalib()
mtx, dist = camera.get_camera_parameters()
test_file = 'camera_cal/calibration1.jpg'
#camera.test_calibration(test_file)

warp = prespectiveTransform()
test_file = 'test_images/test10.jpg'
img = cv2.imread(test_file)
#warp.test_prespective_transform(img)

def overlay_curve(img, edge, lanes, Minv):
    warp_zero = np.zeros_like(edge).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lanes[0].allx, lanes[0].ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lanes[1].allx, lanes[1].ally])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    middle = (lanes[0].allx[-1] + lanes[1].allx[-1])//2
    veh_pos = img.shape[1]//2
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    off_center = (veh_pos - middle) * xm_per_pix # Positive if on right, Negative on left

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curvature (Left)  = %.2f m' % (laneLines[0].radius_of_curvature), (10, 40), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (laneLines[0].radius_of_curvature), (10, 70), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Vehicle position : %.2f m %s of center' % (abs(off_center), 'left' if off_center < 0 else 'right'), (10, 100),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def pipeline(img):
    lane = LaneDetection(camera, warp)
    lane.extract_lane_information(img, useEdge=False, show_images=False)
    #if lane.condition!=1:
    laneLines[0].extract_lane_coordinates(lane.edge_bird_view, True, show_fit=False)
    laneLines[1].extract_lane_coordinates(lane.edge_bird_view, False, show_fit=False)
    result = overlay_curve(img, lane.edge_bird_view, laneLines, warp.mtx_perp_inv)
    return result

def record_result(fin = 'project_video.mp4', fout = 'project_video_out.mp4'):
    clip1 = VideoFileClip(fin)
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(fout, audio=False)

laneLines = [laneInfo(), laneInfo()]

record_result()
#record_result('challenge_video.mp4', 'class_challenge_video1.mp4')

if 0:
    images_path = 'test1/*'
    #images_path = 'under_exposed/*'
    image_files = sorted(glob(images_path))
    laneLines = [laneInfo(), laneInfo()]
    for fname in image_files:
        print(fname)
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = pipeline(img)
        plt.imshow(out)
        plt.show()



