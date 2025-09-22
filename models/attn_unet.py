import tensorflow as tf
from tensorflow.keras import layers, Model
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x
def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1, strides=1, padding="same")(x)
    g_up = layers.UpSampling2D(size=(2,2))(g) if g.shape[1] < theta_x.shape[1] else g
    phi_g = layers.Conv2D(filters, 1, strides=1, padding="same")(g_up)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    out = layers.Multiply()([x, psi])
    return out
def attn_unet_model(input_shape=(128,128,4)):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, 64); p1 = layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 128); p2 = layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 256); p3 = layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 512); p4 = layers.MaxPool2D()(c4)
    b = conv_block(p4, 1024)
    g1 = layers.Conv2D(512, 1, padding="same")(b)
    att1 = attention_gate(c4, g1, 256)
    u1 = layers.UpSampling2D()(b); u1 = layers.Concatenate()([u1, att1]); d1 = conv_block(u1, 512)
    g2 = layers.Conv2D(256, 1, padding="same")(d1)
    att2 = attention_gate(c3, g2, 128)
    u2 = layers.UpSampling2D()(d1); u2 = layers.Concatenate()([u2, att2]); d2 = conv_block(u2, 256)
    g3 = layers.Conv2D(128, 1, padding="same")(d2)
    att3 = attention_gate(c2, g3, 64)
    u3 = layers.UpSampling2D()(d2); u3 = layers.Concatenate()([u3, att3]); d3 = conv_block(u3, 128)
    g4 = layers.Conv2D(64, 1, padding="same")(d3)
    att4 = attention_gate(c1, g4, 32)
    u4 = layers.UpSampling2D()(d3); u4 = layers.Concatenate()([u4, att4]); d4 = conv_block(u4, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return Model(inputs, outputs)
