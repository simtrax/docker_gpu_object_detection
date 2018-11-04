FROM tensorflow/tensorflow:1.10.0-devel-gpu

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

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN echo "export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim" >> ~/.bashrc
RUN python setup.py install

RUN cp /tensorflow/models/research/object_detection/legacy/train.py /tensorflow/models/research/object_detection/
RUN cp /tensorflow/models/research/object_detection/legacy/eval.py /tensorflow/models/research/object_detection/

WORKDIR /

RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools /tensorflow/models/research/

WORKDIR /

RUN git clone https://github.com/simtrax/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 tf_repo

RUN mv tf_repo/xml_to_csv.py /tensorflow/models/research/object_detection/
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

CMD ["echo", "Running tensorflow docker"]