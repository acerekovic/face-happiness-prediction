Face Happiness prediction
====================================

We release two Tensorflow models trained to estimate intensity of a face happiness from an image. One of these models accompanies our submission to the ICMI'16 [2]. We also provide a framework for model training and evaluation. Framework is tested on Ubuntu 14.04 machine equipped with Intel i7-4790K @ 4.00GHz CPU, 16 Gb RAM, and a GeForce GTX 980 Ti GPU, with Tensorflow 0.8.

Models are released for non-commercial research purposes under the [Creative Commons Attribution-NonCommercial License](https://creativecommons.org/licenses/by-nc/4.0/). 

If you find the framework/models to be useful in your research work, a citation to the following paper would be appreciated:

[2]: A. Cerekovic: A deep look into group happiness prediction from images, in Proceedings of the 2016 ACM on International Conference on Multimodal Interaction ICMI'16, preprint

Happy coding (and research)!

Downloading models and prerequisites
------------------------------------
To set up environment, run bash script download_data.sh. The script will download trained models and prerequisites for model training and testing.

Models: VGG16 and GoogLeNet-FC
-------------------------------
Upon running the download_data.sh, the models will be located in the ./data/models/ directory. These models are pre-trained for the task of face happiness intensity prediction. Given the face image, the task is to recognize one of the following happiness intensities:

0 - neutral face
1 - small smile
2 - large smile
3 - small laugh
4 - big laugh
5 - thrilled

Models are trained with the HAPPEI training set [1], and with manually annotated images collected in [Gallager and Chen, 2009][3]. Training dataset contains 3600 images, and is augmented 10 times.

The first model, GoogLeNet-FC model, is introduced in [2]. Given the face image of size 256x256x3, the model is able to estimate the intensity of happiness with 49.1% accuracy. The precision of the model is measured on the HAPPEI validation set, extracted with Face Detection model from [2]. Accuracy of the provided GoogLeNet-FC model is higher than 47% (reported in [2]) because of additional training samples introduced to the HAPPEI training set.

The second model VGG16 has 50.3 % accuracy, measured on the same validation set. Image size for VGG16 is 224x224x3.

The code for loading the GoogLeNet (utils/googlenet_load.py) is modified version of the code from the [TensorBox framework] [4]. The initial weights for the GoogLeNet model also originate from the same source. The code for VGG16 model is modification of [Davi Frossard's code][6]. The code is rewritten and extended for the purpose of training and testing with Tensorflow. 

Demo
------

We provide an example of how model a can be loaded and run. Just run demo.py, with one argument:

```
1. For VGG16 use:
--model='VGG16' 

2. For GoggLeNet-FC use:
--model='GNET'
```

and preview the results. Upon computation, the results will be saved to ./data/results.json.


Training 
------------------
Script train.py serves to train the selected model. The script has to be called with the following arguments:

```
--model="VGG16" (or --model="GNET" for GoogLeNet-FC)
--data_dir="./data/fake_HAPPEI"
```

Where data_dir points to the training and validation sets. The structure of the data_dir directory has to be as follows:

```
--data_dir
    -- \images
    -- data_dir_training.csv
    -- data_dir_validation.csv
```

Images directory contains images, and *.csv files contain a list of image filenames (contained in \images dir) and accompanied labels, such as:
imagefile, label
image1.jpg, 0
image2.jpg, 5

To give oneself a clear overview how data_dir directory should be organized see data/fake_HAPPEI. 

By default, given the input arguments, the framework will augment training dataset 10 times and use undersampling technique to balance the dataset.
Augmentation is done on-the-fly. For data augmentation we use ImageGenerator from Keras.

By default, training is done for 5 epochs. To modify training parameters, optional training arguments are offered, as follows:

```
--weights="./data/model/gnet-model.ckpt-6501"
--num_epochs=10
```

By specifying weights, train.py loads a pretrained model and continues training for a given num_epochs.


Evaluation
----------

Similarily to train.py, evaluate.py should be called with the following params:

```
--model="VGG16" (or --model="VGG16" for GoogLeNet-FC)
--data_dir="./data/fake_HAPPEI"
--weights="./data/models/vgg16.ckpt-6601"
```

Where data_dir points to data_dir directory (see structure in Training paragraph), and weights point to the model that needs to be evaluated.
Evaluation is done in batches, for the sake of the computational power. At the moment, evaluation is done on validation set from the data_dir_validation.csv file. To change it to test set, "testing" parameter should be passed to Loader class from data loader module.

Once evaluation is done, the terminal will output the accuracy result, confusion matrices (normalized and a matrix with a total number of samples), and generate plots.


Tensorboard preview
-------------------

We also provide a possibility to preview training parameters in Tensorboard.
To run the visualization, type in terminal:

```
 tensorboard --logdir='path_to_train_dir'
```
 
Where 'path_to_train_dir' is relative or absolute path to the training directory. By default training directory will be created in /train_dir directory, with a name that ends with date and time when a training started.


References
----------

  [1]: Abhinav Dhall, Roland Goecke, Jyoti Joshi, Jesse Hoey & Tom Gedeon, EmotiW 2016: Video and Group-level Emotion Recognition Challenges, ACM ICMI 2016. link: https://sites.google.com/site/emotiw2016/challenge-details
  
  [2]: A. Cerekovic: A deep look into group happiness prediction from images, in Proceedings of the 2016 ACM on International Conference on Multimodal Interaction ICMI'16, preprint
  [3]: http://chenlab.ece.cornell.edu/people/Andy/Andy_files/cvpr09.pdf
  [4]: https://github.com/Russell91/TensorBox
  [5]: https://creativecommons.org/licenses/by-nc/4.0/
  [6]: http://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
  [7]: https://cs.anu.edu.au/few/Group.htm
