### Literature Review

#### 0x0. Get Start

for two eye gaze problem, my read this article: *Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression*

#### 0x1. Knowledge

1. the gaze directions of two eyes should be consistent physically
2. even if we apply the same regression method, the gaze estimation performance on two eyes can be very different

Hence we need a new strategy that no longer treat both eyes equally.

Strategy: **guide the asymmetric gaze regression by evaluating the performance of the regression strategy w.r.t.different eyes.**



#### 0x2. Main Work

1. Propose a *multi-stream* AR-Net and E-Net.
2. Propose new mechanism of evaluation-guided asymmetric regression.
3. Design ARE-Net



#### 0x3. Two eye asymmetry

Previous work: treat two eyes indiffrently.

Observation: we cannot expect the same accuracy for two eyes, either eye has a chance to be more accurate.

Why asymmetry: head pose, image quality and individuality.

How to solve: propose a network which can tell which eye is of high quality.



#### 0x4. ARE-Net

- AR-Net:
    - it is designed to be able to optimize the two eyes in an asymmetric way
    - structure:
        - the first two streams to extract a 500D deep features from each eye independently, and the last two streams to produce a joint 500D feature in the end
        - input the head pose vector (3D for each eye) before the final regression
        - Base-CNN: similar to AlexNet
    - loss function: weighted angular error
        - The weights λl and λr determine whether the accuracy of the left or the right eye should be considered more important

- E-Net:
    - the evaluation network is trained to predict the probability of the left/right eye image being more efficient in gaze estimation.
    




#### 0x5. My plan

- First try to implement AR-Net, and see the mean err



