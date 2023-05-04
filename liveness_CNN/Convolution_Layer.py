from tensorflow.keras import layers


# stage_name=2,3,4,5; block_name=a,b,c
def ConvBlock(input_tensor, num_output, stride, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=stride, padding='same', name='res' + stage_name + block_name + '_branch2a')(
        input_tensor)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2b')(x)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2b')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2b_relu')(x)

    shortcut = layers.Conv2D(filter2, 1, strides=stride, padding='same',
                             name='res' + stage_name + block_name + '_branch1')(input_tensor)
    shortcut = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch1')(shortcut)

    x = layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_relu')(x)

    return x


def IdentityBlock(input_tensor, num_output, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2a')(
        input_tensor)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2b')(x)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2b')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2b_relu')(x)

    shortcut = input_tensor

    x = layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_relu')(x)

    return x
