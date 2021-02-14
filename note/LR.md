## Literature Review

### 1. Appearance-Based Gaze Estimation in the Wild

#### 0x00 The MPIIGaze Dataset

- 15 participants, 213,659 pictures

- outside of laboratory conditions, i.e during daliy routine
- wilder range of recording location, time, illumination and eye appearance

how to collect: use of laptop application to let volunteers to look at a fixed place, and take pictures of their eyes.

use of laptop to collect: laptops are suited for long-term daily recordings but also because they are an important platform for *eye tracking application*.

#### 0x01 Calibration settings

I think this is used in 3d head pose estimation and face aligment process. I don't use that.

#### 0x02 Method

The CNN is to learn the mapping from *head poses and eye images* to *gaze directions* in the camera coordinate system.

i) Face alignment and 3d head pose estimation

 - detect face
 - generate 6D landmarks

ii) Data normalisation

> first proposed in *Learning-by-Synthesis for Appearance-based 3D Gaze Estimation*

iii) Multimodal CNNs

- 处理输入的2D头部角度`h`和归一化后的眼部图像𝑒，得到最终的2维视线角度向量`g`

- use LeNet

- add `h` in the full connect layer.