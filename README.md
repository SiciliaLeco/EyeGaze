# Eye Gaze Estimation

### 1. Project Overview

This is a research topic in computer vision to predict **where** a person is looking at given the person’s full face. 

Generally, there are two directions of the task:

- 3-D gaze vector estimation is to predict the gaze vector, which is usually used in the automotive safety.
-  2-D gaze position estimation is to predict the horizontal and vertical coordinates on a 2-D screen, which allows utilizing gaze point to control a cursor for human-machine interaction.

Given the training dataset, we can resolve two types of problems: single-eye gaze estimation and two-eye gaze estimation. Apparently, this means our task is to predict one eye gaze direction or both eye. 

Usability: Track the eye movement, provide detailed insights into users' attention.

Challenges: (a) low sensor quality or unknown/challenging environments, and (b) large variations in eye region appearance.

### 2. Related work

#### 2.1 Gaze Estimation Methods

There are two wildly accepted methods for estimating gaze direction: **model-based** and **appearance-based**. Model-based method uses 3D eyeball models and estimate the gaze direction using geometric eye features, while appearance-based method learns generic gaze estimators from large amounts of person, and head pose-independent training data.

Model-based method largely depend on the requirement of external light source to detect eye feature so the modelling process could be a complexing one, and the accuracy for this method is still lower and the robustness is unclear.[1] Appearance-based gaze estimation methods directly use eye images as input and can therefore potentially work with low-resolution eye images. Since the eye images contain many information, so this method needs large amount of data than model-based for the training process.

#### 2.2 Dataset collection 

The Eyediap[2] dataset contains 94 video sequences of 16 participants looking at three different targets. So the gaze direction can be very limited and coarse and can't train a generalized gaze estimator. The UT Multiview[3] dataset collected 50 participants and can be used to sythesise images for head poses. But the problem for these two dataset is that they both record the gaze images under contolled laboratory environment.

The MPIIGaze[1] gaze dataset is used in the task for two reasons: 

- It's recorded outside the lab: when people are at home doing their work, the application on **laptop** capture the images.
- It takes months to record the data, so it contains wider range of recording locations and times, illuminations, and eye appearances.

The MPIIGaze dataset details are shown below:

- 15 participants, 213,659 pictures

- outside of laboratory conditions, i.e during daliy routine
- wilder range of recording location, time, illumination and eye appearance

How to collect: use of laptop application to let volunteers to look at a fixed place, and take pictures of their eyes. (Laptops are suited for long-term daily recordings but also because they are an important platform for *eye tracking application*.)

#### 2.3 Calibration Settings

No matter model-based or appearance-based methods, they both need to collect person-specific data during a calibration step. Previous works on gaze estimation didn't take person-specific caliberation settings into consideration. 

But for the MPIIGaze dataset, since they were collected using different laptops, so the screen size, resolution would be different. Furthermore, the camera coordinate system can also be wide-ranging. What their team did was to obtain the intrinsic parameters for the laptops. In this way, we can add the influence of participant-specific data into our model: 3D positions of each screen plane were estimated using a mirror-based calibration method.

To summarize, MPIIGaze dataset is giving images of the face, calibration settings for a specific participant, 3D gaze vectors of eyes, which is the ground truth for the problem.

### 3. Method

Our task is generally divided into two parts: determine single eye gaze direction for one person, and determine directions for both eyes. Each problem has distinctive method to resolve it.

#### 3.1 Single-eye problem

##### 3.1.1 Problem analysis

For single-eye problem, the overview of the task is to predict a 3D gaze direction for one person, given his face image and head pose information. In the MPIIGaze dataset, the head pose was calculated by the calibration parameters, and the eye is extracted from the face image so we can just focus on the eye image for prediction. 

### 4. Evaluation

### 5. Discussion

### 6. Conclusion



### 7. Timeline

Week 3: Dataset observation, a general understanding of the problem.

Week 4: Determine method to be used, dataset exploration.

Week 5: In-depth learn *Appearanced-based gaze in-the-wild*[1].

Week 6: Implement a multi-modal CNN for single-eye gaze estimation.

Week 7: think of relationship and influence that 3D head pose, 3D gaze vectors. Code to prove the thinking.

Week 8: improve some techniques applied to the data. (Normalize the vector, change of judging metrics) 

Week 9-10: Validation process(K-fold validation), improve the result by tuning hyper-parameters.

Week 11:  start of two-eye mission. Run model on left eye and right ey separately.  

Week 12: Learn Asymmetry technique, Read the paper.[2]

Week 13: implementing AR-Net.

### Appendix

[1] Zhang, Xucong, et al. "Appearance-based gaze estimation in the wild." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2015.

[2] Funes Mora, Kenneth Alberto, Florent Monay, and Jean-Marc Odobez. "Eyediap: A database for the development and evaluation of gaze estimation algorithms from rgb and rgb-d cameras." *Proceedings of the Symposium on Eye Tracking Research and Applications*. 2014.

[3] Sugano, Yusuke, Yasuyuki Matsushita, and Yoichi Sato. "Learning-by-synthesis for appearance-based 3d gaze estimation." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2014.