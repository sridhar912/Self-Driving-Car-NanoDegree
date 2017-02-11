import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
from utils import *

class vehicleDetector:
    def __init__(self,cspace,shape, detect_single_frame=True, display_heat_map = False):
        self.detect_single_frame = detect_single_frame
        if self.detect_single_frame:
            self.nframes = 1
            self.heatmap_thresh = 0
        else:
            self.nframes = 20
            self.heatmap_thresh = 1
        self.frame_cnt = 0
        self.heatmap = np.zeros((shape[0], shape[1], self.nframes), dtype=np.float32)
        self.clf = joblib.load('best.pkl')
        self.scaler = joblib.load('best_scaler.pkl')
        self.clf_thresh = 0.995
        self.y_start_stop_ = [[370, 520], [370, 520], [370, 550], [370, 580]]
        self.x_start_stop_ = [[500, 1100], [500, 1200], [300, None], [280, None]]
        self.scales = [1.125, 1, 0.875, 0.75]
        #self.y_start_stop_ = [[370, 520], [370, 520], [370, 550], [370, 580], [370, 610]]
        #self.x_start_stop_ = [[500, 1100], [500, 1200], [300, None], [280, None], [250, None]]
        #self.scales = [1.125, 1, 0.875, 0.75, 0.625]
        self.result = []
        self.cspace = cspace
        self.edgeThresh = 150
        self.display_heat_map = display_heat_map

    def convert_cspace(self,image):
        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LAB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif self.cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        return feature_image

    def get_window_images(self,img,windows):
        img_windows = []
        for i in range(len(windows)):
            window = windows[i]
            img_windows.append(img[window[0][1]:window[1][1], window[0][0]:window[1][0], :])
        return img_windows

    def detect_vehicle_bbox(self,img_hog,windows,img_windows):
        features = []
        bbox = []
        img_bbox = []
        spatial_features = extract_features_spatial_bin_detect(img_windows, cspace='LAB', spatial_size=(32, 32))
        features.append(spatial_features)
        hist_features = extract_features_color_hist_detect(img_windows, cspace='LAB')
        features.append(hist_features)
        # lbp_features = extract_features_lbp_detect(img_windows)
        # features.append(lbp_features)
        hog_features = extract_features_hog_detect(img_hog[:, :, 0])
        features.append(hog_features)
        feat = np.nan_to_num(np.concatenate(features, axis=1))
        test_features = self.scaler.transform(np.array(feat))
        x = self.clf.predict_proba(test_features)
        x1 = x[:, 1] > self.clf_thresh
        x1 = x1.astype(int)
        ind = np.nonzero(x1)
        for i in range(len(ind[0])):
            bbox.append(windows[ind[0][i]])
            img_bbox.append(img_windows[ind[0][i]])
        return bbox, img_bbox

    def scale_bbox(self,bbox_,scale):
        bbox = []
        for bb1, bb2 in bbox_:
            var1 = np.int(bb1[0] * scale)
            var2 = np.int(bb1[1] * scale)
            var3 = np.int(bb2[0] * scale)
            var4 = np.int(bb2[1] * scale)
            bbox.append(((var1, var2), (var3, var4)))
        return bbox

    def get_bbox_heatmap(self,heat):
        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_thresh[heat > self.heatmap_thresh] = 255
        _, contours, _ = cv2.findContours(np.copy(heat_thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for j, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(((x, y), (x + w, y + h)))
        return boxes

    def evaluate_bbox(self,bbox,images):
        bbox_filter = []
        for ind, image in enumerate(images):
            #image = image * 255
            image = image.astype(np.uint8)
            edge = cv2.Canny(image, 0, 255)
            t = np.nonzero(edge.ravel())
            if len(t[0]) > self.edgeThresh:
                bbox_filter.append(bbox[ind])
        return bbox_filter

    def display_search_region(self,image):
        color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        out =[]
        f, ax = plt.subplots(1, 4, figsize=(12, 12))
        f.tight_layout()
        for indx, scale in enumerate(self.scales):
            img_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            feature_image = self.convert_cspace(img_resized)
            y_start_stop = np.copy(self.y_start_stop_[indx])
            if y_start_stop[0] != None:
                y_start_stop[0] = np.int(np.round(y_start_stop[0] * scale))
            if y_start_stop[1] != None:
                y_start_stop[1] = np.int(np.round(y_start_stop[1] * scale))
            x_start_stop = np.copy(self.x_start_stop_[indx])
            if x_start_stop[0] != None:
                x_start_stop[0] = np.int(np.round(x_start_stop[0] * scale))
            if x_start_stop[1] != None:
                x_start_stop[1] = np.int(np.round(x_start_stop[1] * scale))
            windows = slide_window(feature_image, x_start_stop=x_start_stop, y_start_stop=y_start_stop)
            out = draw_boxes(img_resized,windows,color[indx],2)
            ax[indx%4].imshow(out)
            st = 'Scale:{}'.format(scale)
            ax[indx%4].set_title(st)

        plt.show()

    def process_frame(self,image):
        self.result = []
        draw_image = np.copy(image)
        image = image.astype(np.float32) / 255
        image = cv2.GaussianBlur(image, (5, 5), 0)
        result_bbox = []
        for indx, scale in enumerate(self.scales):
            img_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            feature_image = self.convert_cspace(img_resized)
            y_start_stop = np.copy(self.y_start_stop_[indx])
            if y_start_stop[0] != None:
                y_start_stop[0] = np.int(np.round(y_start_stop[0] * scale))
            if y_start_stop[1] != None:
                y_start_stop[1] = np.int(np.round(y_start_stop[1] * scale))
            x_start_stop = np.copy(self.x_start_stop_[indx])
            if x_start_stop[0] != None:
                x_start_stop[0] = np.int(np.round(x_start_stop[0] * scale))
            if x_start_stop[1] != None:
                x_start_stop[1] = np.int(np.round(x_start_stop[1] * scale))
            windows = slide_window(feature_image, x_start_stop=x_start_stop, y_start_stop=y_start_stop)
            img_windows = self.get_window_images(feature_image,windows)
            roi_hog = feature_image[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]
            bbox_, img_bbox = self.detect_vehicle_bbox(roi_hog,windows,img_windows)
            bbox = self.evaluate_bbox(bbox_,img_bbox)
            bbox_scaled = self.scale_bbox(bbox,(1 / scale))
            result_bbox.extend(bbox_scaled)

        heat = np.zeros(image.shape[:2], dtype=np.float32)
        heat = draw_boxes_heat(heat, result_bbox)
        heat = cv2.GaussianBlur(heat, (15, 15), 0)

        self.heatmap[:, :, self.frame_cnt % self.nframes] = heat

        if self.frame_cnt < self.nframes:
            heat = np.mean(self.heatmap[:, :, :self.frame_cnt + 1], axis=2)
        else:
            heat = np.mean(self.heatmap, axis=2)
        if self.display_heat_map:
            d_image = np.copy(draw_image)
            d_image = draw_boxes(d_image,result_bbox,(255,0,0))
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 12))
            plt.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Input Image')
            ax2.imshow(d_image)
            ax2.set_title('Detections')
            ax3.imshow(heat)
            ax3.set_title('Heat map')
            plt.show()

        bbox_trk = self.get_bbox_heatmap(heat)

        draw_image = draw_boxes(draw_image, bbox_trk)
        self.frame_cnt += 1
        return draw_image
