import numpy as np
import cv2
import matplotlib.pyplot as plt

class prespectiveTransform():

    def __init__(self, src_points = None, dst_points = None):
        if src_points!=None:
            self.src_points = src_points
        else:
            self.src_points = np.float32([[240, 720],
                                          [575, 460],
                                          [715, 460],
                                          [1150, 720]])
        if dst_points!=None:
            self.dst_points = dst_points
        else:
            self.dst_points = np.float32([[440, 720],
                                 [440, 0],
                                 [950, 0],
                                 [950, 720]])

        self.mtx_perp = None
        self.mtx_perp_inv = None

        self.mtx_perp, self.mtx_perp_inv = self.perspective_transform()

    def perspective_transform(self):
        """Compute perspective transform from source and destination points.
        """
        mtx_perp = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        mtx_perp_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        return mtx_perp, mtx_perp_inv

    def warp_image(self, img, inverse=False):
        """Warp an image using a transform matrix.
        """
        img_size = (img.shape[1], img.shape[0])
        if not inverse:
            return cv2.warpPerspective(img, self.mtx_perp, img_size, flags=cv2.INTER_LINEAR)
        else:
            return cv2.warpPerspective(img, self.mtx_perp_inv, img_size, flags=cv2.INTER_LINEAR)

    def get_prespective_parameters(self):
        return self.mtx_perp, self.mtx_perp_inv

    def show_warped_images(self, org, warp):
        """
        Show orginal and undistorted images
        :param org: The original image
        :param undist: The undistorted image
        :return:
        """
        orgr = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
        undistr = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(orgr)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistr)
        ax2.set_title('Warped Image', fontsize=30)
        plt.show()

    def draw_lines(self, img, color=[255, 0, 0], thickness=2):
        """Draw lines on an image.
        """
        for i in range(4):
            x1 = self.src_points[i][0]
            y1 = self.src_points[i][1]
            x2 = self.src_points[(i+1)%4][0]
            y2 = self.src_points[(i+1)%4][1]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def test_prespective_transform(self, img):
        self.draw_lines(img)
        warp = self.warp_image(img)
        self.show_warped_images(img,warp)