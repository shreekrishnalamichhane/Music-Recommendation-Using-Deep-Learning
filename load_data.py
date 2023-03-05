import os
import re
import numpy as np
import cv2
from import_data import create_spectrogram
from slice_spectrogram import slice_spect
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

"""
Converts images and labels into training and testing matrices.
"""


def load_dataset(verbose=0, mode=None, datasetSize=1.0):
    create_spectrogram(verbose, mode)
    slice_spect(verbose, mode)

    # datasetSize is a float value which returns a fraction of the dataset.
    # If set as 1.0 it returns the entire dataset.
    # If set as 0.5 it returns half the dataset.

    if mode == "Train":
        genre = {"Avant-Garde": 0, "International": 1, "Blues": 2, "Jazz": 3, "Classical": 4, "Novelty": 5, "Comedy": 6, "Old-Time / Historic": 7, "Country": 8, "Pop": 9, "Disco": 10, "Rock": 11, "Easy Listening": 12, "Soul-RnB": 13, "Electronic": 14, "Sound Effects": 15, "Folk": 16, "Soundtrack": 17, "Funk": 18, "Spoken": 19, "Hip-Hop": 20, "Audio Collage": 21, "Punk": 22, "Post-Rock": 23, "Lo-Fi": 24, "Field Recordings": 25, "Metal": 26, "Noise": 27, "Psych-Folk": 28, "Krautrock": 29, "Jazz: Vocal": 30, "Experimental": 31, "Electroacoustic": 32, "Ambient Electronic": 33, "Radio Art": 34, "Loud-Rock": 35, "Latin America": 36, "Drone": 37, "Free-Folk": 38, "Noise-Rock": 39, "Psych-Rock": 40, "Bluegrass": 41, "Electro-Punk": 42, "Radio": 43, "Indie-Rock": 44, "Industrial": 45, "No Wave": 46, "Free-Jazz": 47, "Experimental Pop": 48, "French": 49, "Reggae - Dub": 50, "Afrobeat": 51, "Nerdcore": 52, "Garage": 53, "Indian": 54, "New Wave": 55, "Post-Punk": 56, "Sludge": 57, "African": 58, "Freak-Folk": 59, "Jazz: Out": 60, "Progressive": 61, "Alternative Hip-Hop": 62, "Death-Metal": 63, "Middle East": 64, "Singer-Songwriter": 65, "Ambient": 66, "Hardcore": 67, "Power-Pop": 68, "Space-Rock": 69, "Polka": 70, "Balkan": 71, "Unclassifiable": 72, "Europe": 73, "Americana": 74, "Spoken Weird": 75, "Interview": 76, "Black-Metal": 77, "Rockabilly": 78, "Easy Listening: Vocal": 79, "Brazilian": 80, "Asia-Far East": 81, "N. Indian Traditional": 82,
                 "South Indian Traditional": 83, "Bollywood": 84, "Pacific": 85, "Celtic": 86, "Be-Bop": 87, "Big Band/Swing": 88, "British Folk": 89, "Techno": 90, "House": 91, "Glitch": 92, "Minimal Electronic": 93, "Breakcore - Hard": 94, "Sound Poetry": 95, "20th Century Classical": 96, "Poetry": 97, "Talk Radio": 98, "North African": 99, "Sound Collage": 100, "Flamenco": 101, "IDM": 102, "Chiptune": 103, "Musique Concrete": 104, "Improv": 105, "New Age": 106, "Trip-Hop": 107, "Dance": 108, "Chip Music": 109, "Lounge": 110, "Goth": 111, "Composed Music": 112, "Drum & Bass": 113, "Shoegaze": 114, "Kid-Friendly": 115, "Thrash": 116, "Synth Pop": 117, "Banter": 118, "Deep Funk": 119, "Spoken Word": 120, "Chill-out": 121, "Bigbeat": 122, "Surf": 123, "Radio Theater": 124, "Grindcore": 125, "Rock Opera": 126, "Opera": 127, "Chamber Music": 128, "Choral Music": 129, "Symphony": 130, "Minimalism": 131, "Musical Theater": 132, "Dubstep": 133, "Skweee": 134, "Western Swing": 135, "Downtempo": 136, "Cumbia": 137, "Latin": 138, "Sound Art": 139, "Romany (Gypsy)": 140, "Compilation": 141, "Rap": 142, "Breakbeat": 143, "Gospel": 144, "Abstract Hip-Hop": 145, "Reggae - Dancehall": 146, "Spanish": 147, "Country & Western": 148, "Contemporary Classical": 149, "Wonky": 150, "Jungle": 151, "Klezmer": 152, "Holiday": 153, "hiphop": 154, "Salsa": 155, "Nu-Jazz": 156, "Hip-Hop Beats": 157, "Modern Jazz": 158, "Turkish": 159, "Tango": 160, "Fado": 161, "Christmas": 162, "Instrumental": 163}

        if (verbose > 0):
            print("Compiling Training and Testing Sets ...")
        filenames = [os.path.join("Train_Sliced_Images", f) for f in os.listdir("Train_Sliced_Images")
                     if f.endswith(".jpg")]
        images_all = [None]*(len(filenames))
        labels_all = [None]*(len(filenames))
        for f in filenames:
            index = int(
                re.search('Train_Sliced_Images/(.+?)_.*.jpg', f).group(1))
            genre_variable = re.search(
                'Train_Sliced_Images/.*_(.+?).jpg', f).group(1)
            temp = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images_all[index] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            labels_all[index] = genre[genre_variable]

        if (datasetSize == 1.0):
            images = images_all
            labels = labels_all

        else:
            count_max = int(len(images_all)*datasetSize / 8.0)
            count_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            images = []
            labels = []
            for i in range(0, len(images_all)):
                if (count_array[labels_all[i]] < count_max):
                    images.append(images_all[i])
                    labels.append(labels_all[i])
                    count_array[labels_all[i]] += 1
            images = np.array(images)
            labels = np.array(labels)

        images = np.array(images)
        labels = np.array(labels)

        labels = labels.reshape(labels.shape[0], 1)
        train_x, test_x, train_y, test_y = train_test_split(
            images, labels, test_size=0.05, train_size=0.05, shuffle=True)

        # Convert the labels into one-hot vectors.
        train_y = np_utils.to_categorical(train_y)
        test_y = np_utils.to_categorical(test_y, num_classes=164)
        n_classes = len(genre)
        genre_new = {value: key for key, value in genre.items()}

        if os.path.exists('Training_Data'):
            train_x = np.load("Training_Data/train_x.npy")
            train_y = np.load("Training_Data/train_y.npy")
            test_x = np.load("Training_Data/test_x.npy")
            test_y = np.load("Training_Data/test_y.npy")
            return train_x, train_y, test_x, test_y, n_classes, genre_new

        if not os.path.exists('Training_Data'):
            os.makedirs('Training_Data')
        np.save("Training_Data/train_x.npy", train_x)
        np.save("Training_Data/train_y.npy", train_y)
        np.save("Training_Data/test_x.npy", test_x)
        np.save("Training_Data/test_y.npy", test_y)
        return train_x, train_y, test_x, test_y, n_classes, genre_new

    if mode == "Test":
        if (verbose > 0):
            print("Compiling Training and Testing Sets ...")
        filenames = [os.path.join("Test_Sliced_Images", f) for f in os.listdir("Test_Sliced_Images")
                     if f.endswith(".jpg")]
        images = []
        labels = []
        for f in filenames:
            song_variable = re.search(
                'Test_Sliced_Images/.*_(.+?).jpg', f).group(1)
            tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
            labels.append(song_variable)

        images = np.array(images)

        return images, labels
