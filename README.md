## Structure from Motion
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
---

## Overview

Structure from motion (SfM) is the process of estimating the 3-D structure of a scene from a set of 2-D images. SfM is used in many applications, such as 3-D scanning , augmented reality, and visual simultaneous localization and mapping (vSLAM). SfM can be computed in many different ways.

This project presents our pipeline for recreating a 3-D scene using Structure from Motion. Reconstructed a 3 dimensional scene with 2D stereo images from a monocular camera captured from
different views while estimating camera poses along the way. The pipeline consists of Feature Matching, RANSAC Based Outlier feature rejection and Estimation of Fundamental Matrix, Estimation of Essential matrix from F matrix, Camera Pose Estimation and Refinement, Check for Cheirality Condition using Triangulation, Linear and Nonlinear Perspective-n-point estimation, Bundle Adjustment to achieve the results. 

## Technology Used

* Ubuntu 20.04 LTS
* Python Programming Language
* OpenCV Library
* SciPy Library
* Pylint
* Doxygen

## License 

```
MIT License

Copyright (c) 2023 Aditya Jadhav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Demos

- TBD

## Set of Assumptions 

- TBD

## Known Issues/Bugs 

- High BA errors
- 

## Dependencies

- Install OpenCV 3.4.4 or higher and other dependencies.


To Run tests 
```
python test_load_dataset.py
```

## Links

Final Output and Overview --> [Link]