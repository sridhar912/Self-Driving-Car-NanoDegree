import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt

class cameraCalib():

    def __init__(self, calib_image_path = 'camera_cal/'):

        self.mtx = None
        self.dist = None
        self.calib_image_path = calib_image_path
        self.calib_file = self.calib_image_path + 'camera_param.pickle'

        if not os.path.isfile(self.calib_file):
            calib_param = self.calc_calibration_parameters(self.calib_image_path, 6, 9, display_corners=False)
            self.mtx = calib_param['mtx']
            self.dist = calib_param['dist']
            with open(self.calib_file, 'wb') as f:
                pickle.dump(calib_param, file=f)
        else:
            with open(self.calib_file, "rb") as f:
                calib_param = pickle.load(f)
                self.mtx = calib_param['mtx']
                self.dist = calib_param['dist']
                print('Loaded calibration parameters from disk....')


    def calc_calibration_parameters(self, calib_im_path, rows, cols, display_corners=False):
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
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
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

        calib_param = {'mtx': mtx,
                       'dist': dist}
        return calib_param

    def get_camera_parameters(self):
        return self.mtx, self.dist

    def undistort_image(self, img):
        """Undistort an image.
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def show_undistorted_images(self, org, undist):
        """
        Show orginal and undistorted images
        :param org: The original image
        :param undist: The undistorted image
        :return:
        """
        orgr = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
        undistr = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(orgr)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistr)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()

    def test_calibration(self, fname):
        """Test calibration on an example chessboard, and display the result.
        """
        # Load image, draw chessboard and undistort.
        img = cv2.imread(fname)
        img_undist = self.undistort_image(img)
        self.show_undistorted_images(img, img_undist)




