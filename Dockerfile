FROM tensorflow/tensorflow:1.5.0-devel-gpu

# Did not work, seg fault when trying to train
# FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
  git

RUN apt-get install -y protobuf-compiler \
  python-lxml \
  python-pil \
  build-essential cmake pkg-config \
  libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libxvidcore-dev libx264-dev \
  libgtk-3-dev \
  libatlas-base-dev gfortran \
  #python2.7-dev \
  python-dev \
  python-tk \
  vim
  
RUN pip install opencv-python==3.4.0.12 requests
  
# Change to tensorflow dir
WORKDIR /tensorflow

# Clone the models repo
RUN git clone https://github.com/tensorflow/models.git
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
RUN unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# Reset to previous commit
WORKDIR models
RUN git reset --hard 079d67d9a0b3407e8d074a200780f3835413ef99

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc
RUN python setup.py install

# If we reset the repo we do not need to copy this file
# RUN cp /tensorflow/models/research/object_detection/legacy/train.py  /tensorflow/models/research/object_detection/

# Yolo
WORKDIR /

RUN git clone https://github.com/simtrax/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 tf_repo

RUN mv tf_repo/images /tensorflow/models/research/object_detection/
RUN mv tf_repo/training /tensorflow/models/research/object_detection/

RUN mv tf_repo/generate_tfrecord.py /tensorflow/models/research/object_detection/
RUN mv tf_repo/Object_detection_image.py /tensorflow/models/research/object_detection/
RUN mv tf_repo/Object_detection_video.py /tensorflow/models/research/object_detection/
RUN mv tf_repo/Object_detection_webcam.py /tensorflow/models/research/object_detection/

WORKDIR /
# Clean up
RUN rm -r tf_repo

RUN curl -OL http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
RUN tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

RUN mv faster_rcnn_inception_v2_coco_2018_01_28 /tensorflow/models/research/object_detection/

# Clean up
RUN rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

RUN sed -i 's/C:\/tensorflow1/\/tensorflow/g' /tensorflow/models/research/object_detection/training/faster_rcnn_inception_v2_pets.config

CMD ["echo", "Running tensorflow docker"]