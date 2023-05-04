from tensorflow import keras
from tensorflow.keras import layers
from liveness_CNN.Convolution_Layer import ConvBlock
from liveness_CNN.Convolution_Layer import IdentityBlock


def ResNet18(input_shape, class_num):
    input = keras.Input(shape=input_shape, name='input')

    # conv1
    x = layers.Conv2D(64, 7, strides=(2, 2), padding='same', name='conv1')(input)  # 7×7, 64, stride 2
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)  # 3×3 max pool, stride 2
    x = layers.Dropout(0.5)(x)

    # conv2_x
    x = ConvBlock(input_tensor=x, num_output=(64, 64), stride=(1, 1), stage_name='2', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(64, 64), stage_name='2', block_name='b')
    x = layers.Dropout(0.5)(x)

    # conv3_x
    x = ConvBlock(input_tensor=x, num_output=(128, 128), stride=(2, 2), stage_name='3', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(128, 128), stage_name='3', block_name='b')
    x = layers.Dropout(0.5)(x)

    # conv4_x
    x = ConvBlock(input_tensor=x, num_output=(256, 256), stride=(2, 2), stage_name='4', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(256, 256), stage_name='4', block_name='b')
    x = layers.Dropout(0.5)(x)

    # conv5_x
    x = ConvBlock(input_tensor=x, num_output=(512, 512), stride=(2, 2), stage_name='5', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(512, 512), stage_name='5', block_name='b')
    x = layers.Dropout(0.5)(x)

    # average pool, 1000-d fc, softmax
    x = layers.AveragePooling2D((7, 7), strides=(1, 1), name='pool5', padding='same')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(class_num, activation='softmax', name='fc')(x)

    model = keras.Model(input, x, name='resnet18')
    model.summary()
    return model
