# Tensorflow docker container with GPU support
Me trying to get stuff to work..

## Some reading
From guide found [here](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10).

*Each step of training reports the loss. It will start high and get lower and lower as training progresses.*

*For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8. I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are).*

*Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.*

## 1 Build image
Build the image using regular docker build command.

```
docker build -t your_docker_username/prefered_imagename:gpu . --no-cache=true
```

## 2 Run the container
If we don't call /bin/bash the container will exit immediately since there is nothing running.
```
nvidia-docker run -it -v /home/tensorflow-obj/models:/notebooks/models nf/tf:gpu /bin/bash
```

## 3 Prepare for training

When the container is running these commands has to be executed in the models/research folder

```
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
```

### Create TF-record
```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

## 4 Train
cd to the object_detection folder and run this command.
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

### Tensorboard
Open another terminal and open a new bash session in the container.
Run the following command in the objec detection folder
```
tensorboard --logdir=training
```

## 5 Export graph
Replace “XXXX” in “model.ckpt-XXXX” with the highest-numbered .ckpt file in the training folder.

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

## 6 Test detection
Using the file `Object_detection_image.py` one can try detecting objects.  

- Copy a test image to the object_detection folder.  
- Open the file `Object_detection_image.py` with vim. 
- Change NUM_CLASSES variable to the desired number.
- Change the image name to the copied image name.
- At the end of the file change the code to the following:
```
# All the results have been drawn on image. Now display the image.
# cv2.imshow('Object detector', image)

cv2.imwrite('01.png',image)

# Press any key to close the image
# cv2.waitKey(0)

# Clean up
# cv2.destroyAllWindows()
```
- Run the following command:
```
python Object_detection_image.py
```

Now a image should be saved to disk. Copy it out of the container and check it out.

## Useful stuff

### Jupyter

Run with the following command
```
./run_jupyter.sh --allow-root
```

### Copy file from container to host
```
docker cp <containerId>:/file/path/within/container /host/path/target
```
Copy frozen inference graph
```
docker cp <containerId></containerId>:/tensorflow/models/research/object_detection/inference_graph/frozen_inference_graph.pb .
```

### Copy file from host to container
```
docker cp /host/path/target <containerId>:/file/path/within/container
```

### Open ports
If you want to expose tensorboard add this flag
```
-p 6006:6006
```

### Clean up
If you want to remove the container afterwards add this flag
```
--rm
```

## Thanks to
Found the Dockerfile here

[Link](https://github.com/sofwerx/tensorflow-object-detection-docker/tree/master/gpu_docker)