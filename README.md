# ObjectDetectionWithYolov8
Object detection using yolov8 and with kerascv

### Setup
```bash
pip install git+https://github.com/keras-team/keras-cv -q
pip install wget
```
### Load data
Dataset is from [roboflow](https://public.roboflow.com/object-detection/self-driving-car)
```python
path_zip = 'https://public.roboflow.com/ds/6Fnh8IfS02?key=8iDZV73tVg'
path_save = '/content'
wget.download(path_zip, path_save)
!unzip '/content/Self Driving Car.v3-fixed-small.tensorflow.zip' -d '/content/data'
```

### Preprocess
- The data format conforms to the input of Yolov8
```python
def Prepare_data(data):
  gk = data.groupby('filename')
  image_path = []
  boxes = []
  classes_id = []
  for group_name in list(gk.first().index):
    try:
      image_path.append(os.path.join(path_dir , group_name))
    except:
      continue
    group_value = gk.get_group(group_name)
    boxes.append(group_value.loc[: , ['xmin' , 'ymin' , 'xmax' , 'ymax']].values.tolist())
    classes = list(group_value['class'])
    classes_id.append([list(class_mapping.keys())[list(class_mapping.values()).index(cls)] for cls in classes])
  return image_path , boxes , classes_id
```
- Data Augmentation
```python
augument = keras.Sequential(layers=[
    keras_cv.layers.RandomFlip('horizontal' , bounding_box_format= 'xyxy'),
    keras_cv.layers.RandomShear(x_factor = 0.2 , y_factor = 0.2 , bounding_box_format= 'xyxy'),
    keras_cv.layers.JitteredResize(target_size= (640 , 640) , scale_factor= (0.75 , 1.3) , bounding_box_format='xyxy')
])
```
- Visualization
![image1](Data/Screenshot%202024-01-06%20215656.png) 
![image](Data/Screenshot%202024-01-06%20215758.png)

### Creating model
```python
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
yolo = keras_cv.models.YOLOV8Detector(backbone= backbone , num_classes= len(class_mapping) , fpn_depth = 1 , bounding_box_format= 'xyxy')
```
### Training model
```python
yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=[callback],
)
```
### Visualize Predictions
```python
images = next(iter(images_test.take(1)))
y_pred = yolo.predict(images)
y_pred = bounding_box.to_ragged(y_pred)
visualization.plot_bounding_box_gallery(
        images,
        value_range=(0,255),
        rows=2,
        cols=2,
        y_pred=y_pred,
        scale=4,
        font_scale=0.7,
        bounding_box_format='xyxy',
        class_mapping=class_mapping,
        show = True
    )
```
![result](Data/Screenshot%202024-01-06%20220956.png)
![result1](Data/Screenshot%202024-01-06%20220939.png)
![result2](Data/Screenshot%202024-01-06%20221016.png)
![result4](Data/Screenshot%202024-01-06%20221253.png)