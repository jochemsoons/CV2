# Pretrained Models
The Pretrained model folder contains our pre-trained network. In there you
will find two files with D and G in their name, which is for the Generator
and Discriminator, respectively. Load the models using the loadModels
function in the utils file. Check the definition of the saving and loading
function in the utils file for the API call. The api allows you to save both
the optimizer state and the iteration number. When loading the pretrained
model, use only the network checkpoint. Do not use the optimizer states,
since they are different from what you will be using and might lead to bad
convergence.

# Dataset
The data set folder contains the data. Use this dataset to train and test
your GAN Blending part of the assignment. The train and test split is
provided in the respective file. DO NOT TRAIN WITH THE TEST SPLIT.
If you want to create a validation split for yourself, you can further split
the train file to obtain a validation split for yourself.

The data set consists of 4 files for each pair. They are in the following
format:
x_sw_y_z.png   -> The source face crop, pose aligned and expression mapped to
the target
x_fg_z.png     -> The source face to be blended onto the target. This is not
pose or expression mapped.
x_bg_y.png     -> The target face on which the source is to be mapped on.
x_mask_y_z.png -> The full face region mask of the source face crop, that you
want to blend onto the target.


