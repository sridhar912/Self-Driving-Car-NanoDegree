import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
from moviepy.editor import VideoFileClip
import os
import cv2
import pickle
from glob import glob
import matplotlib.image as mpimg
from skimage.exposure import adjust_gamma
from skimage.io import imsave
import math
from skimage.measure import LineModel, ransac

calib_image_path = 'camera_cal/'
calib_file = 'camera_cal/calib_param.pickle'

def calc_calibration_parameters(calib_im_path, rows, cols, display_corners=False):
    """Compute calibration parameters from a set of calibration images.
        Params:
          calib_im_path: Directory of calibration images.
          row: checkerboard row number
          col: checkerboard col number
        Return:
          calib_param = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}
        """
    # Object / image points collections.
    objpoints = []
    imgpoints = []

    # Calibration points from images.
    filenames = os.listdir(calib_im_path)
    for fname in filenames:
        img = cv2.imread(calib_im_path + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Theoretical Grid.
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        # Corners in the image.
        ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            if display_corners:
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                plt.imshow(img)
                plt.show()

        else:
            print('Warning! Not chessboard found in image', fname)
    # Calibration from image points.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       img.shape[0:2],
                                                       None, None)

    calib_param = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}

    return calib_param


def get_camera_calibration(file, calib_image_path):
    """
    Depending on the constant CALC_CAL_POINTS the camera calibration will be
    calculated and stored on disk or loaded.
    """

    if not os.path.isfile(file):
        calib_param = calc_calibration_parameters(calib_image_path, 6, 9, display_corners=False)
        with open(file, 'wb') as f:
            pickle.dump(calib_param, file=f)
    else:
        with open(file, "rb") as f:
            #print('Loaded calibration parameters from disk....')
            calib_param = pickle.load(f)

    return calib_param

def undistort_image(img, mtx, dist):
    """Undistort an image.
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def show_images(img1, img2, title1='', title2='', figsize=(24, 9)):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f.tight_layout()
    ax1.imshow(img1,cmap='gray')
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2,cmap='gray')
    ax2.set_title(title2, fontsize=20)
    #f.show()
    plt.show()

def test_calibration(fname, calib_param):
    """Test calibration on an example chessboard, and display the result.
    """
    # Load image, draw chessboard and undistort.
    img = cv2.imread(fname)
    mtx = calib_param['mtx']
    dist = calib_param['dist']
    print('mtx:',mtx)
    print('dist:', dist)
    undst = undistort_image(img, mtx, dist)
    #show_images(img,undst,'Distorted Image','UnDistorted Image')

def warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR):
    """Warp an image using a transform matrix.
    """
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, mtx_perp, img_size, flags=cv2.INTER_LINEAR)

def perspective_transform(src_points, dst_points):
    """Compute perspective transform from source and destination points.
    """
    mtx_perp = cv2.getPerspectiveTransform(src_points, dst_points)
    mtx_perp_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    return mtx_perp, mtx_perp_inv

test_file = 'camera_cal/calibration1.jpg'
calib_param = get_camera_calibration(calib_file,calib_image_path)
test_calibration(test_file,calib_param)

src_points = np.float32([[240, 720],
            [575, 460],
            [715, 460],
            [1150, 720]])
dst_points = np.float32([[440, 720],
            [440, 0],
            [950, 0],
            [950, 720]])

mtx = calib_param['mtx']
dist = calib_param['dist']

mtx_perp, mtx_perp_inv = perspective_transform(src_points, dst_points)

def extract_color_info(img, threshL=(210,250), threshB=(220,250)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    channelL, channelA, channelB = cv2.split(lab)
    channelL_norm = np.uint8(255 * channelL / np.max(channelL))
    white_lane = cv2.inRange(channelL_norm, threshL[0], threshL[1])
    channelB_norm = np.uint8(255 * channelB / np.max(channelB))
    yellow_lane = cv2.inRange(channelB_norm, threshB[0], threshB[1])
    return white_lane, yellow_lane

def extract_lane_information(img, warp_img, mtx_perp):
    white_lane, yellow_lane = extract_color_info(img)
    color_lane = cv2.bitwise_or(white_lane, yellow_lane)
    color_lane_warped = warp_image(color_lane, mtx_perp, flags=cv2.INTER_LINEAR)
    white_lane_warp, yellow_lane_warp = extract_color_info(warp_img)
    color_lane_warp = cv2.bitwise_or(white_lane_warp, yellow_lane_warp)
    lanes_bird_view = cv2.bitwise_or(color_lane_warp,color_lane_warped)
    lanes_front_view = warp_image(lanes_bird_view, mtx_perp_inv, flags=cv2.INTER_LINEAR)
    is_saturated = check_saturation(white_lane, yellow_lane,white_lane_warp,yellow_lane_warp)
    sobelx = np.absolute(cv2.Sobel(lanes_bird_view, cv2.CV_64F, 1, 0, ksize=3))
    scaled_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    binary_outputx = np.zeros_like(scaled_sobelx)
    binary_outputx[(scaled_sobelx >= 20) & (scaled_sobelx <= 200)] = 1
    show_images(color_lane_warped,color_lane_warp)

    return is_saturated,lanes_front_view, lanes_bird_view

def nonZeroCount(img, offset):
    return cv2.countNonZero(img[offset:,:])

def check_saturation(white_lane, yellow_lane,white_lane_warp,yellow_lane_warp, offset=480, thresh = (500,20000)):
    count_wl = nonZeroCount(white_lane, offset)
    count_wlw = nonZeroCount(white_lane_warp, offset)
    count_yl = nonZeroCount(yellow_lane, offset)
    count_ylw = nonZeroCount(yellow_lane_warp, offset)
    #print(count_wl)
    #print(count_wlw)
    #print(count_yl)
    #print(count_ylw)
    if (count_wl < thresh[1] and count_wlw < thresh[1]) and (count_yl < thresh[1] and count_ylw < thresh[1]):
        if (count_wl < thresh[0] and count_wlw < thresh[0]) or (count_yl < thresh[0] and count_ylw < thresh[0]):
            return True
        else:
            return False
    else:
        return True

def record_result_func1(img,mtx,dist,mtx_perp,mtx_perp_inv):
    img = undistort_image(img, mtx, dist)
    wimg = warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR)
    _, lane_info_front_view, lane_info_bird_view = extract_lane_information(img, wimg, mtx_perp)
    # plt.imshow(lane_info_front_view)
    out = fill_lane(img, wimg, lane_info_bird_view, mtx_perp_inv)
    return out

def record_result_func(img):
    calib_param = get_camera_calibration(calib_file, calib_image_path)
    mtx = calib_param['mtx']
    dist = calib_param['dist']
    src_points = np.float32([[240, 720],
                             [575, 460],
                             [715, 460],
                             [1150, 720]])
    dst_points = np.float32([[440, 720],
                             [440, 0],
                             [950, 0],
                             [950, 720]])
    mtx_perp, mtx_perp_inv = perspective_transform(src_points, dst_points)
    img = cv2.GaussianBlur(img,(5,5),0)
    outimg = record_result_func1(img,mtx,dist,mtx_perp, mtx_perp_inv)
    #color_binary = np.dstack((np.zeros_like(outimg), np.zeros_like(outimg), outimg))
    return outimg

def record_result(fin = 'project_video.mp4', fout = 'output_project_video.mp4'):
    clip1 = VideoFileClip(fin)
    white_clip = clip1.fl_image(record_result_func) #NOTE: this function expects color images!!
    white_clip.write_videofile(fout, audio=False)


def fill_lane(image, wimage, combined_binary, Minv):
    rightx = []
    righty = []
    leftx = []
    lefty = []

    x, y = np.nonzero(np.transpose(combined_binary))
    i = 720
    j = 630
    while j >= 0:
        plt.imshow(combined_binary[j:i, :])
        #plt.interactive(False)
        plt.show(block=False)
        histogram = np.sum(combined_binary[j:i, :], axis=0)
        left_peak = np.argmax(histogram[:640])
        x_idx = np.where((((left_peak - 25) < x) & (x < (left_peak + 25)) & ((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            leftx.extend(x_window.tolist())
            lefty.extend(y_window.tolist())

        right_peak = np.argmax(histogram[640:]) + 640
        x_idx = np.where((((right_peak - 25) < x) & (x < (right_peak + 25)) & ((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            rightx.extend(x_window.tolist())
            righty.extend(y_window.tolist())
        i -= 90
        j -= 90

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]
    rightx_int = right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2]
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_fit[0] * 0 ** 2 + right_fit[1] * 0 + right_fit[2])
    righty = np.append(righty, 0)
    leftx_int = left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2]
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_fit[0] * 0 ** 2 + left_fit[1] * 0 + left_fit[2])
    lefty = np.append(lefty, 0)
    lsort = np.argsort(lefty)
    rsort = np.argsort(righty)
    lefty = lefty[lsort]
    leftx = leftx[lsort]
    righty = righty[rsort]
    rightx = rightx[rsort]
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]

    # Measure Radius of Curvature for each lane line
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(lefty) + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # Calculate the position of the vehicle
    center = abs(640 - ((rightx_int + leftx_int) / 2))

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (combined_binary.shape[1], combined_binary.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(wimage)
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(0, 720)
    ax1.plot(left_fitx, lefty, color='green', linewidth=3)
    ax1.plot(right_fitx, righty, color='green', linewidth=3)
    ax1.set_title('Fit Polynomial to Lane Lines', fontsize=16)
    ax1.invert_yaxis()  # to visualize as we do the images
    ax2.imshow(result)
    ax2.set_title('Fill Lane Between Polynomials', fontsize=16)
    if center < 640:
        ax2.text(200, 100, 'Vehicle is {:.2f}m left of center'.format(center * 3.7 / 700),
                 style='italic', color='white', fontsize=10)
    else:
        ax2.text(200, 100, 'Vehicle is {:.2f}m right of center'.format(center * 3.7 / 700),
                 style='italic', color='white', fontsize=10)
    ax2.text(200, 175, 'Radius of curvature is {}m'.format(int((left_curverad + right_curverad) / 2)),
             style='italic', color='white', fontsize=10)
    #plt.show()
    return result

def show_fit(left_fitx, right_fitx,yvals):
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.plot(right_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis()


def test_polyfit(left_fit,right_fit):
    yvals = np.linspace(0, 720, num=100)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]
    show_fit(left_fitx,right_fitx,yvals)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]+100
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]+100
    show_fit(left_fitx, right_fitx, yvals)
    plt.show()

def test_hist(img, i):
    left_fit = [1.64793383e-04, -4.33532892e-01 , 6.85462321e+02]
    right_fit = [3.27201550e-04, -3.01706242e-01, 9.91134464e+02]
    start = np.int(i * img.shape[0] / 100)
    end = np.int((i + 1) * img.shape[0] / 100)
    for i in range(start, end):
        yvals = i
    left_fitx1 = np.int(left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2] - 50)
    right_fitx1 = np.int(right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2] - 50)
    left_fitx2 = np.int(left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2] + 50)
    right_fitx2 = np.int(right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2] + 50)
    print()

    return left_fitx1, left_fitx2, right_fitx1, right_fitx2


def extract_lane_points(undist, img, Minv):
    right_side_x = []
    right_side_y = []
    left_side_x = []
    left_side_y = []

    past_xcord = 0
    past_ycord = 0
    xycord_ratio = 1
    offset = 50
    r_offset = 20
    ratio = 100

    # right side
    for i in reversed(range(10, 100)):
        histogram = np.sum(img[i * img.shape[0] / ratio:(i + 1) * img.shape[0] / ratio, img.shape[1] / 2:], axis=0)
        #lf1,lf2,rg1, rg2 = test_hist(img, i)
        #histogram1 = np.sum(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, rg1:rg2], axis=0)
        #plt.imshow(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, img.shape[1] / 2:])
        #plt.show()
        #plt.imshow(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, rg1:rg2])
        #plt.show()
        xcord = int(np.argmax(histogram)) + 640
        ycord = int(i * img.shape[0] / ratio)
        if past_ycord > 0:
            xycord_ratio = ((xcord - past_xcord) * (ycord / past_ycord))
        if (ycord == 0 or xcord == 0):
            pass
        elif (abs(xcord - past_xcord) > offset and not (i == 99) and not (past_xcord == 0)):
            pass
        elif (xcord == 640):
            pass
        elif (abs(xycord_ratio) > r_offset):
            pass
        else:
            #print(xycord_ratio)
            #print('Diff X: ', xcord - past_xcord)
            #print('Diff Y: ', ycord - past_ycord)
            right_side_x.append(xcord)
            right_side_y.append(ycord)
            past_xcord = xcord
            past_ycord = ycord
            #print(np.polyfit(right_side_y,right_side_x,1))

    print('Left side')
    past_xcord = 0
    past_ycord = 0
    xycord_ratio = 1
    # left side
    for i in reversed(range(10, 100)):
        #plt.imshow(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, :img.shape[1] / 2])
        #plt.show()
        histogram = np.sum(img[i * img.shape[0] / ratio:(i + 1) * img.shape[0] / ratio, :img.shape[1] / 2], axis=0)
        xcord = int(np.argmax(histogram))
        ycord = int(i * img.shape[0] / ratio)
        if past_ycord > 0:
            xycord_ratio = ((xcord - past_xcord) * (ycord / past_ycord))
        if (ycord == 0 or xcord == 0):
            pass
        elif (abs(xcord - past_xcord) > offset and not (i == 99) and not (past_xcord == 0)):
            pass
        elif (abs(xycord_ratio) > r_offset):
            pass
        else:
            print(xycord_ratio)
            print('Diff X: ', xcord - past_xcord)
            print('Diff Y: ', ycord - past_ycord)
            left_side_x.append(xcord)
            left_side_y.append(ycord)
            past_xcord = xcord
            past_ycord = ycord


    #left_line = (left_side_x, left_side_y)
    #right_line = (right_side_x, right_side_y)
    #left_line = (left_line[0][1:(len(left_line[0]) - 1)], left_line[1][1:(len(left_line[1]) - 1)])
    #right_line = (right_line[0][1:(len(right_line[0]) - 1)], right_line[1][1:(len(right_line[1]) - 1)])
    #left_side_x = left_side_x[1:len(left_side_x)-1]
    #left_side_y = left_side_y[1:len(left_side_y) - 1]
    #right_side_x = right_side_x[1:len(right_side_x) - 1]
    #right_side_y = right_side_y[1:len(right_side_y) - 1]
    left_side_x = np.array(left_side_x).astype(float)
    left_side_y = np.array(left_side_y).astype(float)
    right_side_x = np.array(right_side_x).astype(float)
    right_side_y = np.array(right_side_y).astype(float)

    fit_left = np.polyfit(left_side_y, left_side_x, 2)
    fit_leftx = fit_left[0] * left_side_y ** 2 + fit_left[1] * left_side_y + fit_left[2]
    if len(right_side_y) == 0 | len(right_side_y)==0:
        count = 0
        imsave(
            "/home/sridhar/code/SDCND/ReferencePython/CarND-Advanced-Lane-Lines/Fail/frame_%04d.jpg" % count,
            undist)
    fit_right = np.polyfit(right_side_y, right_side_x, 2)
    fit_rightx = fit_right[0] * right_side_y ** 2 + fit_right[1] * right_side_y + fit_right[2]
    if 1:
        plt.plot(left_side_x, left_side_y, 'o', color='red')
        plt.plot(right_side_x, right_side_y, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(fit_leftx, left_side_y, color='green', linewidth=3)
        plt.plot(fit_rightx, right_side_y, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.show()


    top = 100
    bot = img.shape[0]-1
    topx = fit_left[0] * top ** 2 + fit_left[1] * top + fit_left[2]
    botx = fit_left[0] * bot ** 2 + fit_left[1] * bot + fit_left[2]
    fit_leftx = np.append(fit_leftx,topx)
    fit_leftx = np.append(fit_leftx, botx)
    left_side_y = np.append(left_side_y, top)
    left_side_y = np.append(left_side_y, bot)

    top = 100
    bot = img.shape[0]-1
    topx = fit_right[0] * top ** 2 + fit_right[1] * top + fit_right[2]
    botx = fit_right[0] * bot ** 2 + fit_right[1] * bot + fit_right[2]
    fit_rightx = np.append(fit_rightx, topx)
    fit_rightx = np.append(fit_rightx, botx)
    right_side_y = np.append(right_side_y, top)
    right_side_y = np.append(right_side_y, bot)

    lsort = np.argsort(left_side_y)
    rsort = np.argsort(right_side_y)

    fit_leftx = fit_leftx[lsort]
    left_side_y = left_side_y[lsort]
    fit_rightx = fit_rightx[rsort]
    right_side_y = right_side_y[rsort]

    if 0:
        plt.plot(fit_leftx, left_side_y, 'o', color='red')
        plt.plot(fit_rightx, right_side_y, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(fit_leftx, left_side_y, color='green', linewidth=3)
        plt.plot(fit_rightx, right_side_y, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.show()

    fit_left = np.polyfit(left_side_y, fit_leftx, 2)
    fit_leftx = fit_left[0] * left_side_y ** 2 + fit_left[1] * left_side_y + fit_left[2]
    fit_right = np.polyfit(right_side_y, fit_rightx, 2)
    fit_rightx = fit_right[0] * right_side_y ** 2 + fit_right[1] * right_side_y + fit_right[2]

    #test_polyfit(fit_left,fit_right)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, left_side_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, right_side_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


def extract_lane_curve(left_points, right_points):
    degree_fit = 2

    fit_left = np.polyfit(left_points[1], left_points[0], degree_fit)

    fit_right = np.polyfit(right_points[1], right_points[0], degree_fit)

    x = [x * (3.7 / 700.) for x in left_points[0]]
    y = [x * (30 / 720.) for x in left_points[1]]

    curve = np.polyfit(y, x, degree_fit)

    return fit_left, fit_right, curve

def record_result_f2(img,mtx,dist,mtx_perp,mtx_perp_inv):
    img = undistort_image(img, mtx, dist)
    wimg = warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR)
    is_sat, lane_info_front_view, lane_info_bird_view = extract_lane_information(img, wimg, mtx_perp)
    if not is_sat:
        out = extract_lane_points(img, lane_info_bird_view, mtx_perp_inv)
    else:
        return img
    return out

def record_result_f1(img):
    calib_param = get_camera_calibration(calib_file, calib_image_path)
    mtx = calib_param['mtx']
    dist = calib_param['dist']
    src_points = np.float32([[240, 720],
                             [575, 460],
                             [715, 460],
                             [1150, 720]])
    dst_points = np.float32([[440, 720],
                             [440, 0],
                             [950, 0],
                             [950, 720]])
    mtx_perp, mtx_perp_inv = perspective_transform(src_points, dst_points)
    img = cv2.GaussianBlur(img,(5,5),0)
    outimg = record_result_f2(img,mtx,dist,mtx_perp, mtx_perp_inv)
    #color_binary = np.dstack((np.zeros_like(outimg), np.zeros_like(outimg), outimg))
    return outimg

def record_result_f(fin = 'project_video.mp4', fout = 'output_project_video.mp4'):
    clip1 = VideoFileClip(fin)
    white_clip = clip1.fl_image(record_result_f1) #NOTE: this function expects color images!!
    white_clip.write_videofile(fout, audio=False)

#record_result('challenge_video.mp4', 'challenge_video_out.mp4')
#record_result_f('challenge_video.mp4', 'challenge_video_out.mp4')
#record_result_f()

images_path = 'test/*'
#images_path = 'under_exposed/*'
image_files = glob(images_path)

for fname in image_files:
    x = np.array([1,2,3]).astype(float)
    xx = np.append(x,10)
    print(fname)
    img = undistort_image(cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB), mtx, dist)
    #img = adjust_gamma(img1, 0.4)
    #show_images(img1, img)
    wimg = warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR)
    is_saturated, lane_info_front_view, lane_info_bird_view = extract_lane_information(img, wimg, mtx_perp)
    print(is_saturated)
    mtx_perp, mtx_perp_inv = perspective_transform(src_points, dst_points)
    #wimg_ = warp_image(wimg, mtx_perp_inv, flags=cv2.INTER_LINEAR)
    #plt.imshow(wimg_)
    #plt.show()
    #show_images(lane_info_front_view,lane_info_bird_view)
    if not is_saturated:
        #print()
        sobelx = np.absolute(cv2.Sobel(lane_info_bird_view, cv2.CV_64F, 1, 0, ksize=3))
        scaled_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
        binary_outputx = np.zeros_like(scaled_sobelx)
        binary_outputx[(scaled_sobelx >= 20) & (scaled_sobelx <= 200)] = 1
        out = extract_lane_points(img, binary_outputx, mtx_perp_inv)
        plt.imshow(out)
        plt.show()
    #show_images(img,lane_info_front_view)
    #out = fill_lane(img, wimg,lane_info_bird_view, mtx_perp_inv)
    #find_lane(lane_info_bird_view)

