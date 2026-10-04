import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import os


def train_model():

    # DATASET
    dataset_path = "dataset"

    # IMAGE SETTINGS
    img_size = (224, 224)
    batch_size = 32

    # DATA AUGMENTATION
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # TRAINING DATA
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    # VALIDATION DATA
    validation_data = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    # BASE MODEL
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    # ADD OUR CLASSIFIER
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.3)(x)

    output = Dense(
        len(train_data.class_indices),
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    # COMPILE
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # TRAIN
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=10
    )

    # SAVE MODEL
    os.makedirs("model", exist_ok=True)

    model.save(
        "model/bird_model.keras"
    )

    print("Training Complete!")
    print("Model saved at model/bird_model.keras")

    return True

if __name__ == "__main__":
    train_model()