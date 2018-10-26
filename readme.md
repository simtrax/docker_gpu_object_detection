# Tensorflow docker container with GPU support

Me trying to get stuff to work..

## Build image
Build the image using regular docker build command.

```
docker build -t your_docker_username/prefered_imagename:gpu . --no-cache=true
```

## Run the container
If we don't call /bin/bash the container will exit immediately since there is nothing running.
```
nvidia-docker run -it -v /home/simon/tensorflow-obj/models:/notebooks/models nf/tf:gpu /bin/bash
```

When the container is running these commands has to be executed in the models/research folder

```
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
```

## Start training

Move to the object_detection folder and run this command.
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

## Tensorboard

Run the following command in the objec detection folder
```
tensorboard --logdir=training
```

### Good things to remember

If you want to expose tensorboard add this flag
```
-p 6006:6006
```

If you want to remove the container afterwards add this flag
```
--rm
```

### Potential errors

train.py row 49.
```
from object_detection.legacy import trainer
```

### Thanks to

Found the Dockerfile here

[Link](https://github.com/sofwerx/tensorflow-object-detection-docker/tree/master/gpu_docker)