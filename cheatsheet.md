# ML Workshop | Build Your First Machine Learning Model in Python using Google Colab

The below markdown file consists of commands and code snippets that will help you complete the lab - Build Your First Machine Learning Model in Python using Google Colab.

## Code Snippets

### Access Dataset

```
https://drive.google.com/file/d/1SToG0KxGcMkZFXdpI_KWUWYjrgZcMXRe/view?usp=sharing
```

### Import TensorFlow

```python
import tensorflow as tf
tf.test.gpu_device_name()
```
### Install Keras

```python
!pip install -q keras
```

### Mount Google Drive to Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Accessing dataset from Drive

```python
!unzip '/content/drive/My Drive/fruits.zip'
```

### Labelling the images

```python
from glob import glob

Apple = glob('train/Apple/*.jpg')
Avocado = glob('train/Avocado/*.jpg')
Banana = glob('train/Banana/*.jpg')
Cactus = glob('train/Cactus/*.jpg')
Cherry = glob('train/Cherry/*.jpg')
Dates = glob('train/Dates/*.jpg')
Grape = glob('train/Grape/*.jpg')
Guava = glob('train/Guava/*.jpg')
Kiwi = glob('train/Kiwi/*.jpg')
Lemon = glob('train/Lemon/*.jpg')
Lychee = glob('train/Lychee/*.jpg')
Orange = glob('train/Orange/*.jpg')
Raspberry = glob('train/Raspberry/*.jpg')
Strawberry = glob('train/Strawberry/*.jpg')
Walnut = glob('train/Walnut/*.jpg')

TRAIN_DIR = 'train'
TEST_DIR = 'val'
```
### Considering a model

```python
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input

CLASSES = 15
    
# setup model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
   
# transfer learning
for layer in base_model.layers:
    layer.trainable = False
      
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Data Augmentation

```python
from keras.preprocessing.image import ImageDataGenerator

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')
```

### Training the model

```
EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 320
VALIDATION_STEPS = 64

MODEL_FILE = 'filename.model'

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
model.save(MODEL_FILE)
```

### Defining function for model prediciton

```
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.preprocessing import image
from keras.models import load_model


def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities 
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

labels = ("Apple","Avocado","Banana","Cactus","Cherry","Dates","Grape","Guava","Kiwi","Lemon","Lychee","Orange","Raspberry","Strawberry","Walnut")

```

### Accessing the built model

```python
model = load_model(MODEL_FILE)
```

### Testing the model

```python
img = image.load_img('val/Raspberry/117_100.jpg', target_size=(HEIGHT, WIDTH))
preds = predict(model, img)
j=max(preds)
result = np.where(preds == j)
index_val = result[0][0]
prediction = labels[index_val]
print("The predicted fruit is:",prediction)
```