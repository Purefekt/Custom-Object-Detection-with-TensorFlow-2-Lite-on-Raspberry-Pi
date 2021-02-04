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
Use the following command to export the model to TensorFlow Lite
```
python export_tflite_graph_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --output_directory exported-models\my_tflite_model
```
