import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import adjust_gamma

NORMAL = 1
DARK = 2
BRIGHT = 3

class LaneDetection():

    def __init__(self, cameraInfo, prespectiveInfo):

        self.edge_bird_view = None
        self.edge_front_view = None
        self.img_undist = None
        self.img_undist_warp = None
        self.cameraInfo = cameraInfo
        self.prespectiveInfo = prespectiveInfo

        self.mtx, self.dist = cameraInfo.get_camera_parameters()

        self.mtx_perp, self.mtx_perp_inv = prespectiveInfo.get_prespective_parameters()

        # 1--> normal 2--> Dark 3-->bright
        self.condition = NORMAL

    def nonZeroCount(self, img, offset):
        return cv2.countNonZero(img[offset:, :])

    def check_saturation(self, white_lane, yellow_lane, white_lane_warp, yellow_lane_warp, offset=480, thresh=(500, 20000)):
        count_wl = self.nonZeroCount(white_lane, offset)
        count_wlw = self.nonZeroCount(white_lane_warp, offset)
        count_yl = self.nonZeroCount(yellow_lane, offset)
        count_ylw = self.nonZeroCount(yellow_lane_warp, offset)
        if (count_wl < thresh[1] and count_wlw < thresh[1]):
            if (count_wl < thresh[0] and count_wlw < thresh[0]) or (count_yl < thresh[0] and count_ylw < thresh[0]) or (
                    count_yl > thresh[1] or count_ylw > thresh[1]):
                return DARK
            else:
                return NORMAL
        else:
            return BRIGHT

    def extract_color_info(self, img, threshL=(210, 250), threshB=(200, 250)):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
        channelL, channelA, channelB = cv2.split(lab)
        channelL_norm = np.uint8(255 * channelL / np.max(channelL))
        white_lane = cv2.inRange(channelL_norm, threshL[0], threshL[1])
        channelB_norm = np.uint8(255 * channelB / np.max(channelB))
        yellow_lane = cv2.inRange(channelB_norm, threshB[0], threshB[1])
        #plt.imshow(channelL_norm)
        #plt.show()
        return white_lane, yellow_lane

    def extract_sobel_edge(self,img):
        sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
        sobel_output = np.zeros_like(scaled_sobel)
        sobel_output[(scaled_sobel >= 20) & (scaled_sobel <= 200)] = 255
        return sobel_output

    def extract_lane_information_diff_condition(self, img, condition):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
        if condition == 2:
            gray_norm = adjust_gamma(gray, 0.4)
        else:
            gray_norm = adjust_gamma(gray, 5)
        #gray_norm = np.uint8(255 * (gray) / np.max(gray))

        sobelx = np.absolute(cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=15))
        sobely = np.absolute(cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=15))
        scaled_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
        binary_outputx = np.zeros_like(scaled_sobelx)
        binary_outputx[(scaled_sobelx >= 20) & (scaled_sobelx <= 200)] = 1
        scaled_sobely = np.uint8(255 * sobely / np.max(sobely))
        binary_outputy = np.zeros_like(scaled_sobely)
        binary_outputy[(scaled_sobely >= 20) & (scaled_sobely <= 200)] = 1
        # show_images(binary_outputx,binary_outputy)
        absgraddir = np.arctan2((binary_outputy), (binary_outputx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= 0.7) & (absgraddir <= 0.8)] = 1
        lanes_front_view = np.uint8(255 * binary_output / np.max(binary_output))
        lanes_bird_view = self.prespectiveInfo.warp_image(lanes_front_view)
        return lanes_front_view, lanes_bird_view


    def extract_lane_information(self, img, useEdge = True, show_images = False):
        img_undist = self.cameraInfo.undistort_image(img)
        img_undist_warp = self.prespectiveInfo.warp_image(img_undist)
        white_lane, yellow_lane = self.extract_color_info(img_undist)
        color_lane = cv2.bitwise_or(white_lane, yellow_lane)
        color_lane_warped = self.prespectiveInfo.warp_image(color_lane)
        white_lane_warp, yellow_lane_warp = self.extract_color_info(img_undist_warp)
        color_lane_warp = cv2.bitwise_or(white_lane_warp, yellow_lane_warp)
        lanes_bird_view = cv2.bitwise_or(color_lane_warp, color_lane_warped)
        lanes_front_view = self.prespectiveInfo.warp_image(lanes_bird_view,inverse=True)
        condition = self.check_saturation(white_lane, yellow_lane, white_lane_warp, yellow_lane_warp)

        if condition != 1:
            # Currently not used
            #print()
            lanes_front_view, lanes_bird_view = self.extract_lane_information_diff_condition(img_undist, condition)
        if useEdge:
            edge_front_view = self.extract_sobel_edge(lanes_front_view)
            edge_bird_view = self.extract_sobel_edge(lanes_bird_view)
            self.edge_bird_view = edge_bird_view
            self.edge_front_view = edge_front_view
        else:
            self.edge_bird_view = lanes_bird_view
            self.edge_front_view = lanes_front_view
            self.img_undist = img_undist

        if show_images:
            self.show_output(img_undist,white_lane,yellow_lane)
            self.show_output(img_undist_warp,white_lane_warp, yellow_lane_warp)
            self.show_output(img_undist,self.edge_front_view,self.edge_bird_view,'Input','Combined Front View','Combined BirdEye View')


        self.img_undist = img_undist
        self.img_undist_warp = img_undist_warp
        self.condition = condition

    def show_output(self, img1, img2, img3, t1 = 'Input', t2 = 'White Lane', t3 = 'Yellow Lane'):
        """
        Show orginal and undistorted images
        :param org: The original image
        :param undist: The undistorted image
        :return:
        """
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(img1)
        ax1.set_title(t1, fontsize=20)
        ax2.imshow(img2, cmap ='gray')
        ax2.set_title(t2, fontsize=20)
        ax3.imshow(img3, cmap='gray')
        ax3.set_title(t3, fontsize=20)
        plt.show()

    def get_undistored_image(self):
        return self.img_undist

    def get_warped_image(self):
        return self.img_undist_warp

    def get_lane_output(self):
        return self.edge_front_view, self.edge_bird_view



