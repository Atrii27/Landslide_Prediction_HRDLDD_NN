import tensorflow as tf
from tensorflow.keras import layers, Model
def res_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding="same")(x)
    y = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)
    return y
def encoder_block(x, filters):
    y = res_block(x, filters)
    p = layers.MaxPool2D()(y)
    return y, p
def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = res_block(x, filters)
    return x
def resunet_model(input_shape=(128,128,4)):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)
    b = res_block(p4, 1024)
    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return Model(inputs, outputs)
