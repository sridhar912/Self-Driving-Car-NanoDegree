import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.metrics import mean_absolute_error

class laneInfo():
    def __init__(self, offset = 100, points_ratio = 100, top_ypoint = 0):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial equation for the best fit
        self.best_fit_eq = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # polynomial equation for the most recent fit
        self.current_fit_eq = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []
        # offset between previous and current coordinates
        self.offset = offset
        # Number of points to extract in line ratio
        self.ratio = points_ratio
        # extrapolate top point
        self.top_y_point = top_ypoint
        # meters per pixel in y dimension
        self.ym_per_pix = 30 / 720
        # meteres per pixel in x dimension
        self.xm_per_pix = 3.7 / 700
        # Number of frames for full search
        self.frames_confidence = 10
        # Track frame count
        self.frame_count = 0
        # Number of points to get confidence
        self.min_points_count = 5
        # Search whole image or region
        self.usePrev = False
        # Search restricted region offset
        self.restricted_offset = 50
        # Fit error threshold from previous frams
        self.error_fit = 200
        # Flag to indicate previous or current result used for detection
        self.prevResult = False
        self.current_err = 0

    def get_restricted_search(self,starty,endy):
        for i in range(starty, endy):
            yvals = i
        startx = np.int(self.best_fit[0] * yvals ** 2 + self.best_fit[1] * yvals + self.best_fit[2] - self.restricted_offset)
        endx = np.int(self.best_fit[0] * yvals ** 2 + self.best_fit[1] * yvals + self.best_fit[2] + self.restricted_offset)
        return startx, endx


    def extract_lane_coordinates(self, img, getLeft, show_fit = False):

        past_xcord = 0
        past_ycord = 0
        allx = []
        ally = []

        for i in reversed(range(1, 99)):
            starty = np.int(i * img.shape[0] / self.ratio)
            endy = np.int((i + 1) * img.shape[0] / self.ratio)
            xcord = 0
            ycord = 0
            if not getLeft:
                if not self.usePrev:  # Search whole image
                    startx = np.int(img.shape[1] / 2)
                    endx = np.int(img.shape[1])
                    histogram = np.sum(img[starty:endy, startx:endx], axis=0)
                    if len(histogram) > 0:
                        xcord = int(np.argmax(histogram)) + 640
                else:
                    startx, endx = self.get_restricted_search(starty,endy)
                    histogram = np.sum(img[starty:endy, startx:endx], axis=0)
                    #plt.imshow(img[starty:endy, startx:endx])
                    #plt.show()
                    if len(histogram) > 0:
                        xcord = int(np.argmax(histogram)) + startx
            else:
                if not self.usePrev:  # Search whole image
                    startx = 0
                    endx = np.int(img.shape[1] / 2)
                    histogram = np.sum(img[starty:endy, startx:endx], axis=0)
                    if len(histogram) > 0:
                        xcord = int(np.argmax(histogram))
                else:
                    startx, endx = self.get_restricted_search(starty,endy)
                    histogram = np.sum(img[starty:endy, startx:endx], axis=0)
                    if len(histogram) > 0:
                        xcord = int(np.argmax(histogram)) + startx

            # lf1,lf2,rg1, rg2 = test_hist(img, i)
            # histogram1 = np.sum(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, rg1:rg2], axis=0)
            #plt.imshow(img[start:end, :half])
            #plt.show()
            #plt.plot(histogram)
            #plt.show()

            ycord = int(i * img.shape[0] / self.ratio)
            if (ycord == 0 or xcord == 0):
                pass
            elif (abs(xcord - past_xcord) > self.offset and not (i == 99) and not (past_xcord == 0)):
                pass
            elif (xcord == 640) or (xcord == startx):
                pass
            else:
                # print(xycord_ratio)
                # print('Diff X: ', xcord - past_xcord)
                # print('Diff Y: ', ycord - past_ycord)
                #print(xcord)
                #print(ycord)
                allx.append(xcord)
                ally.append(ycord)
                past_xcord = xcord
                past_ycord = ycord

        allx = np.array(allx).astype(float)
        ally = np.array(ally).astype(float)

        if len(allx) > 0 or len(ally) > 0:
            allx = self.fit_line_poly(allx, ally)
            bottom = img.shape[0] - 1
            allx, ally = self.extrapolate_point(bottom, allx, ally)
            allx = self.fit_line_poly(allx, ally)
            self.frame_count += 1
            if self.frame_count > self.frames_confidence:
                self.frame_count = self.frames_confidence
                self.usePrev = True
            if not self.evaluate_current_fit():
                self.best_fit = self.current_fit
                self.best_fit_eq = self.current_fit_eq
                self.allx = np.array(allx).astype(float)
                self.ally = np.array(ally).astype(float)
            else:
                self.best_fit = self.best_fit
                self.best_fit_eq = self.best_fit_eq
        else:
            # Use previous frame
            allx = self.allx
            ally = self.ally
            self.frame_count -= 1
            if self.frame_count < 0:
                self.usePrev = False

        if show_fit:
            self.show_line_fit(allx, ally)

        self.radius_of_curvature = self.get_curvature_radius()

    def evaluate_current_fit(self):
        self.prevResult = False
        if self.best_fit == None:
            return self.prevResult
        y = np.linspace(1, 50, num=50)
        x1 = self.current_fit_eq(y)
        x2 = self.best_fit_eq(y)
        err = mean_absolute_error(x1,x2)
        if err > self.error_fit:
            self.prevResult =  True
        self.current_err = err
        return self.prevResult


    def fit_line_poly(self,x,y):
        fit = np.polyfit(y, x, 2)
        xcord_fit = fit[0] * y ** 2 + fit[1] * y + fit[2]
        self.current_fit = fit
        self.current_fit_eq = np.poly1d(fit)
        return xcord_fit

    def show_line_fit(self,x, y):
        plt.plot(x, y, 'o', color='red')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(x, y, color='green', linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.show()

    def extrapolate_point(self, bot, x, y):
        fit = self.current_fit
        top = self.top_y_point
        topx = fit[0] * top ** 2 + fit[1] * top + fit[2]
        botx = fit[0] * bot ** 2 + fit[1] * bot + fit[2]
        x = np.append(x, topx)
        x = np.append(x, botx)
        y = np.append(y, top)
        y = np.append(y, bot)
        # Sort the element so that they are in order
        index = np.argsort(y)
        x = x[index]
        y = y[index]
        return x, y

    def get_curvature_radius(self):
        curve = np.polyfit(self.ym_per_pix * self.ally, self.xm_per_pix * self.allx, 2)

        rad_cur = ((1 + (2 * curve[0] * np.max(self.ally) + curve[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * curve[0])

        return rad_cur
