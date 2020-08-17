/* installing requirements */
pip install -r /content/requirements.txt 

/* Import */
import numpy as np
import cv2
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout, Flatten
from tf.keras.layers import Conv2D
from tf.keras.optimizers import Adam
from tf.keras.layers import MaxPooling2D
from tf.keras.preprocessing.image import ImageDataGenerator

/* Unzip the dataset */
!unzip /content/786787_1351797_bundle_archive.zip

/* Initialising training and validation generators */
train_dir = 'train'
val_dir = 'test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
        )
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
        )  
       
       
/* Convolution network building */
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

/* Training dataset */
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=28709 // 64,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=7178 // 64)
model.save_weights('model.h5')

/*  */
