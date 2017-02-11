---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![cars][output/cars.png]
![noncars][output/non_cars.png]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LAB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hogcars][output/hog_cars.png]
![hognoncars][output/hog_noncars.png]

The code for this implemenation can be seen in the notebook cellnumber 3 and 4

#### 2. Explain how you settled on your final choice of HOG parameters.
Testing was done for different combination of `HOG` like `orientation = (9,18)`, `cellsize = (8,12,16)` and `block_size = (2,3)`. Since the length of the feature vector increases when changing from default configuration `(Orientation=9, cellsize = 8, block_size = 2`, the dafault configuration was selected. Even though  certain parameters performed better in terms on accuracy on validation set, keeping in mind about real time detection criteria, the default configuration was chosen. Also, HOG was tested for different color channel and found that `LAB` color space was better than the rest.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
In order to come up with the best parameters, different combination of `HOG`, `spatial bin`, `color histogram` and also `LBP (Local Binary Patern)` were tested. Also, classifier like `SVC (Linear kernel with probability ), LinearSVC and MLP` were used to analysis the performance. A sample subset result is shown in the tabular column below.
![Analysis][output/experiment.png]
Based on the experiment, `MLP classifier` was chosen since it gave better detection as well the best runtime.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In order to make the classifier detect vehicles which are off different sizes, there is a need to build `image pyramid`. The scales chosen for building the pyramids are `[1.125, 1, 0.875, 0.75]`. Since the model size was kept at (64,64), some far vehicles were not detected. In order to compensate for this, image was upscaled by 1.125 factor. As we know that at the far region in the image around 30-50m, vehicle size will be around `60-80` pixels where at the near region, the vehicle size will increse and will be around `200-250` pixel. This observed was used to design the search region. A the far region, the serach region window was small whereas it is bigger in the near region. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As stated previously, the model used spatial bin, color hist as well HOG feature with MLP as the classifier. In order to speed by the execution, rather than taking features for cropped image, hog was computed for the entire image and feature for the entire image was combined into a single vector and then classifier was used to predict the output for the entire image in a single go. This drastically improved the execution time. `Current run time of the algorithm stands at an average 0.5 second per frame`.

The implementation can be in the function `process_frame()` in `vehicleDetection.py`

![result][output/result.png]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/BuWwSEln250)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `cv2.findContours` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Also, to eliminate some false coming on the road, number of edge count in the bounding box was used to eliminate false which falls on the road plain surfaces `self.edgeThresh`

Here's an example result showing the heatmap from a series of frames of video, the result of `cv2.findContours` and the bounding boxes then overlaid on the image.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code does not work for the different overlap ratio. This is due to extraction of HOG for entire image. To avoid this, hardcoded values for stride was used in HOG calculation. If this can fixed, better accuracy can be achieved since right now only overlap raio of 0.5 was used. The final bounding box is bit shaky and does not properly fit the vehicle. Post processing can be done to fix this using edge criteria. The solution uses list based variable for input and output. If array was used with pre-allocated memory, then runtime can be further reduced. Currently, this implementation, does not take care of detecting smaller vehicles. This can be improved by using smaller model like (32,32) etc. Also rather than building the image pyramid, it is possible to build model pyramid which can further speed up the detection. Reference for model pyramid implementation can be seen [here](https://www.robots.ox.ac.uk/~vgg/rg/papers/DollarBMVC10FPDW.pdf). 

