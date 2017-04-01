import csv
import argparse
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def generator(samples, batch_size=256):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                center_image = cv2.imread(center_path)
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
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def main():
    parser = argparse.ArgumentParser(description='Record Path')
    parser.add_argument(
        'path',
        type=str,
        help='Path that contains IMG/ and driving_log.csv',
        nargs='+'
    )

    args = parser.parse_args()

    samples = []
    for path in args.path:
        log_path  = os.path.join(path, 'driving_log.csv')
        imgs_path = os.path.join(path, 'IMG')

        samples += parse_csv(log_path, imgs_path)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    model = define_model(input_shape=(160,320,3))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=5)

    model.save('model.h5')

if __name__ == '__main__':
    main()

