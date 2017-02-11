import numpy as np
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_feature_lbp(image,radius=3,n_points = 24):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(img, n_points, radius)
    return lbp.ravel()

def extract_features_spatial_bin_detect(images, cspace='RGB', spatial_size=(32, 32)):
    # Create a list to append feature vectors to
    spatial_features = []
    # apply color conversion if other than 'RGB'
    for image in images:
        # Apply bin_spatial() to get spatial color features
        spatial_features.append(bin_spatial(image, size=spatial_size))
        # Return list of feature vectors
    return spatial_features

def extract_features_lbp_detect(images,radius=3,n_points = 24):
    lbp_features = []
    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(img, n_points, radius)
        lbp_features.append(lbp.ravel())
    return lbp_features

def extract_features_color_hist_detect(images, cspace='RGB', hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    hist_features = []
    # apply color conversion if other than 'RGB'
    for image in images:
        # Apply color_hist() also with a color space option now
        hist_features.append(color_hist(image, nbins=hist_bins, bins_range=hist_range))
    # Return list of feature vectors
    return hist_features

def extract_features_hog_detect(img,orient=9, pix_per_cell=8, cell_per_block=2):
    n_cells_x = int(np.floor(64 // pix_per_cell))
    n_cells_y = int(np.floor(64 // pix_per_cell))
    n_blocks_x = (n_cells_x - cell_per_block) + 1
    n_blocks_y = (n_cells_y - cell_per_block) + 1
    hog_img_features= hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                              feature_vector=False)
    sy, sx = img.shape
    cx, cy = (pix_per_cell,pix_per_cell)
    bx, by = (cell_per_block,cell_per_block)
    n_cellsx = int(sx // cx)  # number of cells in x
    n_cellsy = int(sy // cy)  # number of cells in y
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1

    hog_feature = []
    # This step is fixed for overlap of 0.5. if you change the overlap, this needs to be changed
    step_x = 4
    step_y = 4
    block_y = 0
    while block_y + n_blocks_y <= n_blocksy:
        block_x = 0
        while block_x + n_blocks_x <= n_blocksx:
            hog_feature.append(hog_img_features[block_y:block_y+n_blocks_y,block_x:block_x+n_blocks_x,:,:,:].ravel())
            block_x += step_x
        block_y += step_y

    return hog_feature

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features_spatial_bin(image, cspace='RGB', spatial_size=(32, 32)):
    # Create a list to append feature vectors to
    features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LAB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else:
        feature_image = np.copy(image)
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Return list of feature vectors
    return spatial_features

def extract_features_color_hist(image, cspace='RGB', hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LAB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(image)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # Return list of feature vectors
    return hist_features

def extract_features_hog(image, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,vis=False, feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LAB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis, feature_vec))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis, feature_vec)
    # Return list of feature vectors
    return hog_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, sb_cspace='RGB', spatial_size=(32, 32),
                     ht_cspace='RGB', hist_bins=32, hist_range=(0, 256),hg_cspace='RGB',
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     useSpatialFeat = True, useHistFeat = True, useHogFeat = True, useLBP = True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        temp_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # Apply bin_spatial() to get spatial color features
        if useSpatialFeat:
            spatial_features = extract_features_spatial_bin(image,sb_cspace,spatial_size)
            temp_features.append(spatial_features)
        # Apply color_hist() also with a color space option now
        if useHistFeat:
            hist_features = extract_features_color_hist(image,ht_cspace,hist_bins,hist_range)
            temp_features.append(hist_features)
        # Call get_hog_features() with vis=False, feature_vec=True
        if useHogFeat:
            hog_features = extract_features_hog(image,hg_cspace,orient,pix_per_cell,cell_per_block,hog_channel)
            temp_features.append(hog_features)
        if useLBP:
            lbp_feature = extract_feature_lbp(image)
            temp_features.append(lbp_feature)
        # Append the new feature vector to the features list
        features.append(np.concatenate(temp_features))
    # Return list of feature vectors
    return features

def extract_features_detection(image, sb_cspace='RGB', spatial_size=(32, 32),
                     ht_cspace='RGB', hist_bins=32, hist_range=(0, 256),hg_cspace='RGB',
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     useSpatialFeat = True, useHistFeat = True, useHogFeat = True, useLBP = True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    temp_features = []
    # Apply bin_spatial() to get spatial color features
    if useSpatialFeat:
        spatial_features = extract_features_spatial_bin(image,sb_cspace,spatial_size)
        temp_features.append(spatial_features)
    # Apply color_hist() also with a color space option now
    if useHistFeat:
        hist_features = extract_features_color_hist(image,ht_cspace,hist_bins,hist_range)
        temp_features.append(hist_features)
    # Call get_hog_features() with vis=False, feature_vec=True
    if useHogFeat:
        hog_features = extract_features_hog(image,hg_cspace,orient,pix_per_cell,cell_per_block,hog_channel)
        temp_features.append(hog_features)
    if useLBP:
        lbp_feature = extract_feature_lbp(image)
        temp_features.append(lbp_feature)
    # Append the new feature vector to the features list
    features.append(np.concatenate(temp_features))
    # Return list of feature vectors
    return features

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(255, 0, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_boxes_heat(img, bboxes):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bb in bboxes:
        # Draw a rectangle given bbox coordinates
        imcopy[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]] += 1
    # Return the image copy with boxes drawn
    return imcopy