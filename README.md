# Custom Object Detection with TensorFlow 2 Lite on Raspberry Pi
This repository continues from my [last project](https://github.com/Purefekt/Custom-Object-Detection-with-TensorFlow-2) where i built a custom object detector for my face using TensorFlow 2. Now i will export that model to TensorFlow 2 Lite so that i can run it on a raspberry pi 4.  
I had saved an unexported copy of my trained model which i will use to export to TensorFlow Lite.  
I will use the same Tensorflow folder which i had created in the last project, it contains all the important scripts.
## Exporting the model
Open the Anaconda terminal and activate the virtual environment
```
conda activate tensorflow
```
Changing directories
```
cd C:\TensorFlow\workspace\training_demo
```
Use the following command to export the model.
```
python export_tflite_graph_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --output_directory exported-models\my_tflite_model
```
## Creating a New Environment and Installing TensorFlow Nightly
To avoid version conflicts, i created a new Anaconda virtual environment to hold all the packages necessary for conversion. First, i will deactivate the current environment with
```
conda deactivate
```
This command will create a new environment for TFLite conversion.
```
conda create -n tflite pip python=3.7
```
Using this command we can see all virtual environments.
```
conda info --envs
```
To activate the TFLite environment
```
conda activate tflite
```
**Note: This virtual environment must be activated everytime the anaconda terminal is closed**  
Now i will install TensorFlow in this virtual environment. However, in this environment i will not just be installing standard TensorFlow. i will be installing tf-nightly. This package is a nightly updated build of TensorFlow. This means it contains the very latest features that TensorFlow has to offer. I will be installing the CPU version. To start the installation.
```
pip install tf-nightly
```
Sanity check to test the installation.
```
python
```
```
>>> import tensorflow as tf
>>> print(tf.__version__)
```
Correct installation gives the following output.
```
2.5.0-dev20210203
```
## Converting the model to TensorFlow Lite
Cd into the training_demo directory.
```
cd C:\TensorFlow\workspace\training_demo
```
The following command converts the model to TensorFlow Lite.
```
python convert-to-tflite.py
```
File called ```model.tflite``` should appear in the directory ```exported-models\my_tflite_model\saved_model```
**Add image here**
