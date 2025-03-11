import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras as keras
from keras.src.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler =ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,patience=5,verbose=1, min_lr=1e-6
)

early_stopping = EarlyStopping(
    monitor='val_loss',       # metoda de monitorizare
    patience=10,               # nr de epoci care nu s-au imbunatațit inainte de oprire
    restore_best_weights=True # intoarce ponderile cele mai bune
)
checkpoint = ModelCheckpoint('unet_brain_segmentation.keras', save_best_only=True)

def loadImage(image_path, target_size=(256, 256)):
    images = []
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])  # Ensure sorted order
    for img in image_files:
        img_path = os.path.join(image_path, img)  # join path
        img = load_img(img_path, color_mode="grayscale")
        img = img.resize(target_size)
        img = img_to_array(img) / 255.0  # Normalize image
        images.append(img)
    return np.array(images).reshape(-1, target_size[0], target_size[1], 1)

def loadMask(mask_path, target_size=(256, 256)):
    masks = []
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.jpg')])  # Ensure sorted order
    for mask in mask_files:
        mask_path_full = os.path.join(mask_path, mask)  # join path
        mask = load_img(mask_path_full, color_mode="grayscale")
        mask = mask.resize(target_size)
        mask = img_to_array(mask) / 255.0  # Normalize mask
        masks.append(mask)
    return np.array(masks).reshape(-1, target_size[0], target_size[1], 1)  # Ensure masks are (256, 256, 1)

def encoder_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)

    x_shape = tf.keras.backend.int_shape(x)[1:3]
    skip_features = tf.keras.layers.Resizing(x_shape[0], x_shape[1])(skip_features)
    x = tf.keras.layers.Concatenate()([x, skip_features])

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def unet_model(input_shape=(256, 256, 1), num_classes=1):
    inputs = tf.keras.layers.Input(input_shape)

    s1 = encoder_block(inputs, 32)
    s2 = encoder_block(s1, 64)
    s3 = encoder_block(s2, 128)
    s4 = encoder_block(s3, 256)

    b1 = tf.keras.layers.Conv2D(512, 3, padding='same')(s4)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(512, 3, padding='same')(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)

    s5 = decoder_block(b1, s4, 256)
    s6 = decoder_block(s5, s3, 128)
    s7 = decoder_block(s6, s2, 64)
    s8 = decoder_block(s7, s1, 32)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(s8)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    return model

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    bce = keras.losses.BinaryCrossentropy()(y_true, y_pred)  # pierdere BCE
    smooth = 1e-6  # evita divizarea la zero
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 0.5 * bce + 0.5 * (1 - dice)  # pierderea combinata

@tf.keras.utils.register_keras_serializable()
def dice_score(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


if __name__ == '__main__':
    image_dir = r"D:/WORK/SET DE DATE/SetdeDate1/RMN"
    mask_dir = r"D:/WORK/SET DE DATE/SetdeDate1/Mask"
    images = loadImage(image_dir)
    masks = loadMask(mask_dir)

    model = unet_model(input_shape=(256, 256, 1))
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_score])  # Folosește dice_loss
    model.summary()

    history = model.fit(images, masks, batch_size=4, epochs=50, validation_split=0.2, callbacks=[early_stopping,lr_scheduler])

    model.save('unet_brain_segmentation.keras')
    model = tf.keras.models.load_model(r"D:/WORK/PI-P.V2/unet_brain_tumor_segmentation.keras", custom_objects={"dice_loss": dice_loss, "dice_score": dice_score})

    imageTest_path = r"D:/WORK/SET DE DATE/SetDateTest/RMN-TEST"
    maskTest_path = r"D:/WORK/SET DE DATE/SetDateTest/MASK-TEST/MASK-TEST"
    testImg = loadImage(imageTest_path)
    testMask = loadMask(maskTest_path)

    hist = model.evaluate(testImg, testMask, 8)
    print(f"Loss on test set: {hist[0]}\n")
    print(f"Accuracy on test set: {hist[1]}\n")

    #plt.plot(history.history['loss'], label='Train loss')
    #plt.plot(history.history['val_loss'], label='Validation loss')
    #plt.legend()
    #plt.show()

    prediction = model.predict(testImg, 8)

    for i in range(len(testImg)):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(testImg[i], cmap='gray')
        plt.title('Imagine originala')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(testMask[i], cmap='gray')
        plt.title('Masca reala')
        plt.axis('off')

        predicted_mask = (prediction[i].squeeze() > 0.3).astype(np.uint8)
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Masca prezisa')
        plt.axis('off')

        plt.show()
