# C++ Final Project: "Place Recognition using Bag of Visual Words"

Welcome to my implementation of the C++ Final Project. This readme outlines the steps needed to get the project up and running and also discusses certain nuances of the way things are structured.

## Prerequisites

**Boost** and **OpenCV4** are the only two prerequisites. The respective installation instructions can be found [here](https://docs.opencv.org/4.3.0/d7/d9f/tutorial_linux_install.html) and [here](https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html). Please make sure to also install OpenCV's extra modules.

The project was tested using OpenCV version 4.3 and Boost version 1.66.0.

## How to start?

Assuming you have already downloaded the latest artifacts, simply head to the directory named `bin` and look for the file called `main`. This is the only file you will need to run the project (with it's default settings).

The program takes a number of command line arguments, described as below, to set various parameters that tune the functions to your liking and also optionally allows you to simply fix certain parameters in a configuration file to make things a little less tedious.

```
General Options:
  -h [ --help ]                         display help message
  -v [ --verbose ]                      print verbose output
  -c [ --config-file ] arg              path to the configuration file
  -I [ --image-path ] arg               path to image dataset
  -D [ --descriptor-path ] arg          path to precomputed feature descriptors
  -H [ --histogram-path ] arg           path to precomputed image histograms
  -Q [ --query-path ] arg               path to query image(s)

Configuration Options:
  --use-flann arg                       use FLANN for histogram computations
                                        (default true)
  --use-opencv-kmeans arg               use opencv kmeans implementation
                                        (default true)
  -k [ --num-clusters ] arg             number of clusters
                                        (default 100)
  -m [ --max-iter ] arg                 maximum number of iterations
                                        (default 25)
  -e [ --epsilon ] arg                  stop iterations if specified accuracy, 
                                        epsilon, is reached
                                        (default 1e-6)
  -n [ --num-similar ] arg              number of similar images to find
                                        (default 10)
  --reweight arg                        perform TF-IDF reweighting for 
                                        histograms
                                        (default false)
  --save-histograms arg                 save histogram dataset to disk
                                        (default true)
  --save-descriptors arg                save descriptors dataset to disk
                                        (default false)
  -Q [ --query-path ] arg               path to query image(s)
```

All configuration options can also be specified on the command line, in which case the values specified on the command line will take preference over those in the configuration file.

A sample configuration file, named `bow_params.cfg` can also be found under the `bin` directory.

## Dataset Directory Structure

The program assumes a certain directory structure for the dataset that it uses to store/load files and complains otherwise.

```
<dataset_root_dir>
├── <any_name>      # Directory where the (png) image dataset is stored
├── descriptors     # Directory where the descriptor dataset is stored
└── histograms      # Directory where the histogram dataset is stored
    ├── codebook    # The computed codebook too is stored in this directory
    └── idf         # And so are the inverse document frequencies, if any
```

Note that the descriptor and histogram files are stored with the same name as the original image.
