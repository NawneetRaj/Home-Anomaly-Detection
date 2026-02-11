from keras.models import Sequential
from keras.layers import Conv2D

def build_csrnet(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(1, (1,1), activation='linear', padding='same'))
    return model
