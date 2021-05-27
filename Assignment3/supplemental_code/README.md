# CV2 assignment3 part 4.2

README FOR ASSIGNMENT PART 4.2

**AUTHORS**:

Jochem Soons    -   11327030
Jeroen van Wely -   11289988
Niek IJzerman   -   11318740

This repository contains the code for assignment 3: 2D-to-3D for FaceSwapping
The assignment is part the course CV2 of the Master programme Artificial Intelligence at the University of Amsterdam.

**FILES INCLUDED**

- ./data:

  - Contains all used input images for running our experiments, default usage of images is set in the main.py file, but of course you are free to change the input images in checking results.
- sec_4_2.py

  - Contains all code for the questions in section 4.2.:
    - 4.2.1 Morphable model
    - 4.2.2 Pinhole camera model
    - 4.2.3 Latent parameters estimation
    - 4.2.4 Texturing
    - 4.2.5 Energy optimization using multiple frames
    - 4.2.6 Face Swapping
- supplemental_code.py

  - Contains the code provided to us (e.g. the render() function), and we also added code for setting a seed and loading + dumping files via Pickle
- plot.py

  - Contains all code for plotting results in Matplotlib.
- main.py:
  Our main file which can be executed to produce results for one of the questions of the assignment.

**INSTRUCTIONS OF USAGE**

To reproduce the results displayed in our report, please run 'python3 main.py'.

You will than be asked to give a question number as input. You can choose to type in:

- 4.2.1:
  - you will then run the experiments that are displayed in section 2.2.1 of our report (sampling objects).
- 4.2.2:
  - you will then run the experiments that are displayed in section 2.2.2 of our report (rotating/translating objects and landmarks).
- 4.2.3:
  - you will then run the experiments that are displayed in section 2.2.3 of our report (latent parameter estimation).
- 4.2.4:
  - you will then run the experiments that are displayed in section 2.2.4 of our report (computing transformed mesh and extracting texture)
- 4.2.5:
  - you will then run the experiments that are displayed in section 2.2.5 of our report (optimization on multiple frames).
- 4.2.6:
  - you will then run the experiments that are displayed in section 2.2.6 of our report (face swapping on a pair of images).

**REQUIREMENTS**

As stated in the general README, we used a conda environment for running our code, that we exported as yml file: refer to the README in the parent folder for more information.
