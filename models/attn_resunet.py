# attn_resunet_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
# Residual Convolutional Block
def res_conv_block(x, filters):
    x_skip = x
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Residual connection
    if x_skip.shape[-1] != filters:
        x_skip = layers.Conv2D(filters, (1,1), padding='same')(x_skip)
    x = layers.Add()([x, x_skip])
    x = layers.Activation('relu')(x)
    return x
# Attention Gate
def attention_gate(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, (1,1), padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, (1,1), padding='same')(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, (1,1), padding='same')(act)
    psi = layers.Activation('sigmoid')(psi)
    out = layers.Multiply()([x, psi])
    return out
# Encoder & Decoder Blocks
def encoder_block(x, filters):
    x = res_conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(x)
    return x, p
def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding='same')(x)
    skip = attention_gate(skip, x, filters // 2)
    x = layers.Concatenate()([x, skip])
    x = res_conv_block(x, filters)
    return x
# Attention Residual U-Net
def build_attn_resunet(input_shape=(128,128,3), n_classes=1):
    inputs = layers.Input(input_shape)
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    # Bridge
    b1 = res_conv_block(p4, 1024)
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    # Output
    outputs = layers.Conv2D(n_classes, (1,1), activation='sigmoid')(d4
    model = models.Model(inputs, outputs, name="Attention_ResUNet")
    return model

