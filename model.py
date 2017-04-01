import csv
import argparse
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

def parse_csv(csv_file, img_path):
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        try:
            measurement = float(line[3])
            measurements.append(measurement)
        except ValueError:
            print("***Couldn't parse line, skipping***")
            continue

        center_path = line[0]
        filename = os.path.basename(center_path)
        curr_path = os.path.join(img_path, filename)
        image = cv2.imread(curr_path)
        images.append(image)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return (X_train, y_train)

def main():
    parser = argparse.ArgumentParser(description='Record Path')
    parser.add_argument(
        'path',
        type=str,
        help='Path that contains IMG/ and driving_log.csv'
    )

    args = parser.parse_args()

    print(os.path.join(args.path, 'IMG'))
    (X_train, y_train) = parse_csv(os.path.join(args.path, 'driving_log.csv'),
                                   os.path.join(args.path, 'IMG'))

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

    model.save('model.h5')

if __name__ == '__main__':
    main()

