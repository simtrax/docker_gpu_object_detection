# Tensorflow docker container with GPU support
Me trying out object detection in various ways..

## 1 Build docker image
Build the image using regular docker build command.

```
docker build -t niftitech/tensorflow:gpu . --no-cache=true
```

## 2 Run the container
If we don't call /bin/bash the container will exit immediately since there is nothing running.

We share our images and the training folder.
```
nvidia-docker run -it -v `pwd`/images:/tensorflow/models/research/object_detection/images -v `pwd`/training:/tensorflow/models/research/object_detection/training -v `pwd`/:/shared_with_host -p 6006:6006 niftitech/tensorflow:gpu /bin/bash
```

## 3 Get started
When the container is running these commands has to be executed in the `models/research` folder

```
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
```

## 4 Generate csv files
From inside the `object_detection` folder run

```
python xml_to_csv.py
```

## 5 Generate TF-record
Edit the (host file) `generate_tfrecord.py` so that the labels are correct.
I edited `num_examples` to be equal to the total number test.record items.

Copy following file inside the container

```
cp /shared_with_host/generate_tfrecord.py .
```

Then run 

```
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

## 6 Train
Update .config file for the model with the correct number of examples.

    num_classes: NUMBER_OF_CLASSES
    num_examples: NUMBER_OF_TEST_IMAGES

Run this command.

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
- Change `NUM_CLASSES` variable to the desired number.
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

Add the following parameter to `visualize_boxes_and_labels_on_image_array` method to change how many boxes should be drawn

```
max_boxes_to_draw=100
```

- Run the following command:
```
python Object_detection_image.py
```

Now a image should be saved to disk. Copy it out of the container and check it out.

## 7 Save boxes to a file

Add this code to the end of the `Object_detection_image.py` file.

```
height, width, channels = image.shape

tempBoxes = np.squeeze(boxes)
tempScores = np.squeeze(scores)
tempClasses = np.squeeze(classes).astype(np.int32)

file = open("results.txt", "w")

file.write("left,right,top,bottom,score,class\n")

for i in range(min(100, tempBoxes.shape[0])):
    if scores is None or tempScores[i] > 0.80:
        box = tuple(tempBoxes[i].tolist())
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * width, xmax * width,
                                    ymin * height, ymax * height)

        score = str(int(100*tempScores[i]))
        class_name = category_index[tempClasses[i]]['name']

        file.write(str(left) + "," + str(right) + "," + str(top) + "," + str(bottom) + "," + score  + "," + str(class_name) + "\n")
```