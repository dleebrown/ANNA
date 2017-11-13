# ANNA - Artificial Neural Network Abundances

Work in progress code to automatically parameterize stellar spectra using a convolutional neural network. Designed to be trained on synthetic stellar spectra with optional data augmentation including continuum error, radial velocity, and instrumental signal-to-noise profile. After training, ANNA can infer a user-specified number of atmospheric parameters from input spectra of the same form as the training data. 

ANNA is currently in a "beta" state. It has been successfully used to classify stellar spectra, and demonstrated to return robust results under a variety of use conditions. The tutorial included in the repository has been tested on both macOS and Linux (with and without a GPU). However, the source code hasn't been fully de-spaghettified, the documentation is a little thin, and there may be significant revisions to the code and/or functionality of the program in the future. 

If you are interested in using ANNA, look for Lee-Brown et. al (2018, in prep), which will discuss the performance of ANNA under a range of use conditions and demonstrate its capabilities. If you are not interested in waiting that long or have questions about ANNA that aren't answered in the manual, feel free to get in touch with me. 

# Author
Donald Lee-Brown ([@dleebrown](https://github.com/dleebrown))

# Acknowledgement
This work was partially supported by NSF grant AST-1211621. 
