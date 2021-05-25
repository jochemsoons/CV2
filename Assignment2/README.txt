/// INFORMATION & AUTHORS ///

Authors:
Jochem Soons    -   11327030
Jeroen van Wely -   11289988
Niek IJzerman   -   11318740

This repository contains the code for assignment 2: Structure from Motion.
The assignment is part the course CV2 of the Master programme Artificial Intelligence at the University of Amsterdam.


/// FILES INCLUDED ///

- Data/House/
    - The House .png FILES

- fundamental_matrix.py:
    Contains the code to compute the fundamental matrix F using the eight point algorithm

- RANSAC.py:
    Contains the code of our RANSAC implementation which can be optionally used in fundamental_matrix.py

- ICP.py:
    Contains the code of the iterative closest point algorithm we implemented for assignment 1.

- point_view_matrix.py:
    Contains the code to produce, plot and save our point_view_matrix.

- helpers.py:
    Contains helper function used in our pipeline, such as functions to detect and match interest points

- main.py:
    Our main file which can be executed to produce results for one of the questions of the assignment.


/// INSTRUCTIONS OF USAGE ///

To reproduce the results displayed in our report, please run 'python3 main.py'. 

You will than be asked to give a question number as input. You can choose to type in:

- 3.4:      you will then run the experiments that are displayed in section 3 of our report (plotting the epipolar lines).
- 4.2:      you will then run the experiments that are displayed in section 4 of our report (creating the point view matrices).
- 5.2:      you will then run the experiments that are displayed in section 5 of our report (plotting structure obtained through SFM).
- 6:        you will then run the experiments that are displayed in section 6 of our report (Additional work: eliminating affine ambiguity)
- plots:    you will then run additional plots that are displayed in section 3 of our report (the line plots)

/// REQUIREMENTS ///

We used a conda environment for running our code, that we exported as yml file: see environment.yml.

To create the environment, run:

    conda env create -f environment.yml

To activate the environment, run:

    conda activate CV2_ass_2

///////////////////////////////////////////////////