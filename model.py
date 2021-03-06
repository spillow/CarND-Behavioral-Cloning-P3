import csv
import argparse
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array, load_img

def generator(samples, batch_size=256):
    """Read center camera and steering wheel angle data
       from logs in `batch_size` batches."""
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                center_image = load_img(center_path)
                center_image = img_to_array(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def parse_csv(csv_file, imgs_path):
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            new_path = os.path.join(imgs_path, os.path.basename(line[0]))
            line[0] = new_path
            lines.append(line)

    return lines

def define_model(input_shape):
    model = Sequential()
    # Normalize the image data to the range [-0.5, 0.5].
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    # 1x1x3 layer to learn color space converter.
    model.add(Convolution2D(3,1,1))
    # Sequence of convolutional layers followed by max pooling to help
    # curb overfitting.
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(20,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(40,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(80,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Steering wheel output
    model.add(Dense(1))

    return model

def main():
    parser = argparse.ArgumentParser(description='Record Path')
    parser.add_argument(
        'path',
        type=str,
        help='Path(s) that contains IMG/ and driving_log.csv',
        nargs='+'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Path to keras model to output as image for network visualization'
    )

    args = parser.parse_args()

    samples = []
    for path in args.path:
        # Examine all paths provided and concatenate them into one big log.
        log_path  = os.path.join(path, 'driving_log.csv')
        imgs_path = os.path.join(path, 'IMG')

        samples += parse_csv(log_path, imgs_path)

    # randomly break the camera image samples into a training and validation set
    # with an 80%/20% split.
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # use a generator to feed the image data to the GPU in batches since we don't have
    # enough GPU memory to fit the whole data set in one shot (run on GTX 970 with 4GB).
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    model = define_model(input_shape=(160,320,3))

    # This is a regression network.  MSE is an appropriate loss function as opposed
    # to cross entropy as used in the previous traffic sign classifier.
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=5)

    model.save('model.h5')

    if args.visualize:
        # must have pydot_ng and graphviz installed for this to work.
        from keras.utils.visualize_util import plot
        plot(model, to_file='model.png', show_shapes=True)

if __name__ == '__main__':
    main()

