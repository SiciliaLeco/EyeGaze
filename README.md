# Gaze Estimation

### 1. Project Overview

This is a task to predict **where** a person is looking at given the person’s full face. 

Two directions of task:

- 3-D gaze vector estimation is to predict the gaze vector, which is usually used in the automotive safety.
-  2-D gaze position estimation is to predict the horizontal and vertical coordinates on a 2-D screen, which allows utilizing gaze point to control a cursor for human-machine interaction.

usability: track the eye movement, provide detailed insights into users' attention.

challenges: (a) low sensor quality or unknown/challenging environments, and (b) large variations in eye region appearance.

notes here:https://github.com/SiciliaLeco/EyeGaze/tree/master/note


### 2. timeline

Week 3: Dataset observation, a general understanding of the problem.

Week 4: Determine method to be used, dataset exploration.

Week 5: In-depth learn *Appearanced-based gaze in-the-wild*[1].

Week 6: Implement a multi-modal CNN for single-eye gaze estimation.

Week 7-8: improve some techniques applied to the data. (Normalize the vector, change of judging metrics) 

Week 9-10: Validation process(K-fold validation), improve the result by tuning hyper-parameters.

Week 11:  start of two-eye mission. Run model on left eye and right ey separately.  

Week 12: Learn Asymmetry technique, Read the paper.[2]

Week 13: implementing AR-Net.
