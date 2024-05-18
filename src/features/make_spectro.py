


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os, json, math, librosa

import IPython.display as ipd
import librosa.display

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

import sklearn.model_selection as sk

from sklearn.model_selection import train_test_split

import imageio.v3 as iio
import importlib
import librosa
from librosa.core import time_to_samples
import librosa.display as lplt
import pandas as pd
import csv



def add_line_to_csv(filename, data):
    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        # Write the new line of data
        csv_writer.writerow(data)


def make_png(dirname, filename):
    try:
        y, sr = librosa.load(dirname + filename)
        

        end_sample = time_to_samples(30, sr=sr)
        y = y[:end_sample]

        hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(abs(D))
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', vmin=-30, vmax=45)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')


        filename = filename[:-4] + ".png"

        # Define the path where you want to save the figure
        save_path = '/Users/kailaiwang/Documents/CS4701/my_spectro/' + filename

        # Save the figure with the specific name
        plt.savefig(save_path)

        # Display the plot
        plt.close()
    except Exception as e:
        print(f"Error processing {filename}: {e}")


def make_30(dirname, filename):
    import numpy as np
    y, sr = librosa.load(dirname + filename)

    end_sample = time_to_samples(30, sr=sr)
    y = y[:end_sample]

    hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(abs(D))
    
    # 1: Chroma

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    #print(np.shape(chroma)) # (12,1293)

    chroma_stft_mean = np.mean(np.mean(chroma, axis=1))
    print(chroma_stft_mean)

    chroma_stft_var = np.mean(np.var(chroma, axis=1))
    # print(np.shape(chroma_stft_var))
    print(chroma_stft_var)

    import numpy as np

    # Calculate the magnitude spectrogram
    D = np.abs(librosa.stft(y))

    # Compute the RMS values for each frame
    rms = librosa.feature.rms(S=D)

    # Calculate the mean and variance of the RMS values
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    print("RMS Mean:", rms_mean)
    print("RMS Variance:", rms_var)

    D = librosa.stft(y)

    # Calculate Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(D), sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroids)
    spectral_centroid_var = np.var(spectral_centroids)

    # Calculate Spectral Bandwidth
    spectral_bandwidths = librosa.feature.spectral_bandwidth(S=np.abs(D), sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidths)
    spectral_bandwidth_var = np.var(spectral_bandwidths)

    print("Spectral Centroid Mean:", spectral_centroid_mean)
    print("Spectral Centroid Variance:", spectral_centroid_var)
    print("Spectral Bandwidth Mean:", spectral_bandwidth_mean)
    print("Spectral Bandwidth Variance:", spectral_bandwidth_var)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    print(rolloff_mean)
    print(rolloff_var)


    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    print(zero_crossing_rate_mean)
    print(zero_crossing_rate_var)


    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmony_mean = np.mean(y_harmonic)
    harmony_var = np.var(y_harmonic)
    print(harmony_mean)
    print(harmony_var)


    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    perceptr_mean = np.mean(mfccs)
    perceptr_var = np.var(mfccs)
    print(perceptr_mean)
    print(perceptr_var)


    tempo, _ = librosa.beat.beat_track(y=y, sr = sr)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Calculate mean and variance for each MFCC
    mfcc_means = np.mean(mfccs, axis=1)  # Compute mean along each MFCC
    mfcc_vars = np.var(mfccs, axis=1)    # Compute variance along each MFCC

    # Extract individual means and variances for each MFCC
    mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean, mfcc6_mean, mfcc7_mean, mfcc8_mean, mfcc9_mean, mfcc10_mean, mfcc11_mean, mfcc12_mean, mfcc13_mean, mfcc14_mean, mfcc15_mean, mfcc16_mean, mfcc17_mean, mfcc18_mean, mfcc19_mean, mfcc20_mean = mfcc_means

    mfcc1_var, mfcc2_var, mfcc3_var, mfcc4_var, mfcc5_var, mfcc6_var, mfcc7_var, mfcc8_var, mfcc9_var, mfcc10_var, mfcc11_var, mfcc12_var, mfcc13_var, mfcc14_var, mfcc15_var, mfcc16_var, mfcc17_var, mfcc18_var, mfcc19_var, mfcc20_var = mfcc_vars


    # Example usage
    data = {
            "id": [filename],
            "chroma_stft_mean": [chroma_stft_mean],
            "chroma_stft_var": [chroma_stft_var],
            "rms_mean": [rms_mean],
            "rms_var": [rms_var],
            "spectral_centroid_mean": [spectral_centroid_mean],
            "spectral_centroid_var": [spectral_centroid_var],
            "spectral_bandwidth_mean": [spectral_bandwidth_mean],
            "spectral_bandwidth_var": [spectral_bandwidth_var],
            "rolloff_mean": [rolloff_mean],
            "rolloff_var": [rolloff_var],
            "zero_crossing_rate_mean": [zero_crossing_rate_mean],
            "zero_crossing_rate_var": [zero_crossing_rate_var],
            "harmony_mean": [harmony_mean],
            "harmony_var": [harmony_var],
            "perceptr_mean": [perceptr_mean],
            "perceptr_var": [perceptr_var],
            "tempo": [tempo],
            "mfcc1_mean": [mfcc1_mean],
            "mfcc1_var": [mfcc1_var],
            "mfcc2_mean": [mfcc2_mean],
            "mfcc2_var": [mfcc2_var],
            "mfcc3_mean": [mfcc3_mean],
            "mfcc3_var": [mfcc3_var],
            "mfcc4_mean": [mfcc4_mean],
            "mfcc4_var": [mfcc4_var],
            "mfcc5_mean": [mfcc5_mean],
            "mfcc5_var": [mfcc5_var],
            "mfcc6_mean": [mfcc6_mean],
            "mfcc6_var": [mfcc6_var],
            "mfcc7_mean": [mfcc7_mean],
            "mfcc7_var": [mfcc7_var],
            "mfcc8_mean": [mfcc8_mean],
            "mfcc8_var": [mfcc8_var],
            "mfcc9_mean": [mfcc9_mean],
            "mfcc9_var": [mfcc9_var],
            "mfcc10_mean": [mfcc10_mean],
            "mfcc10_var": [mfcc10_var],
            "mfcc11_mean": [mfcc11_mean],
            "mfcc11_var": [mfcc11_var],
            "mfcc12_mean": [mfcc12_mean],
            "mfcc12_var": [mfcc12_var],
            "mfcc13_mean": [mfcc13_mean],
            "mfcc13_var": [mfcc13_var],
            "mfcc14_mean": [mfcc14_mean],
            "mfcc14_var": [mfcc14_var],
            "mfcc15_mean": [mfcc15_mean],
            "mfcc15_var": [mfcc15_var],
            "mfcc16_mean": [mfcc16_mean],
            "mfcc16_var": [mfcc16_var],
            "mfcc17_mean": [mfcc17_mean],
            "mfcc17_var": [mfcc17_var],
            "mfcc18_mean": [mfcc18_mean],
            "mfcc18_var": [mfcc18_var],
            "mfcc19_mean": [mfcc19_mean],
            "mfcc19_var": [mfcc19_var],
            "mfcc20_mean": [mfcc20_mean],
            "mfcc20_var": [mfcc20_var]
        }


    filename = '/Users/kailaiwang/Documents/CS4701/my_csv/my_csv.csv'

    # Extract the list of floats from the data dictionary
    new_data = [value[0] for value in data.values()]

    # Append the new data to the CSV file
    add_line_to_csv(filename, new_data)




def make_3(dirname, filename):
    import numpy as np

    
    for index in range(0,30,3):
        try:
            print("working on index", index)


            y, sr = librosa.load(dirname + filename)
            start_sample = time_to_samples(index,sr=sr)
            end_sample = time_to_samples(index+3, sr=sr)
            y = y[start_sample:end_sample]

            if y.size == 0:
                print(index, index+3)

            hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

            D = librosa.stft(y)
            D_db = librosa.amplitude_to_db(abs(D))
            
            # 1: Chroma

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            #print(np.shape(chroma)) # (12,1293)

            chroma_stft_mean = np.mean(np.mean(chroma, axis=1))
            #print(chroma_stft_mean)

            chroma_stft_var = np.mean(np.var(chroma, axis=1))
            # print(np.shape(chroma_stft_var))
            #print(chroma_stft_var)

            import numpy as np

            # Calculate the magnitude spectrogram
            D = np.abs(librosa.stft(y))

            # Compute the RMS values for each frame
            rms = librosa.feature.rms(S=D)

            # Calculate the mean and variance of the RMS values
            rms_mean = np.mean(rms)
            rms_var = np.var(rms)

            #print("RMS Mean:", rms_mean)
            #print("RMS Variance:", rms_var)

            D = librosa.stft(y)

            # Calculate Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(D), sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroids)
            spectral_centroid_var = np.var(spectral_centroids)

            # Calculate Spectral Bandwidth
            spectral_bandwidths = librosa.feature.spectral_bandwidth(S=np.abs(D), sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidths)
            spectral_bandwidth_var = np.var(spectral_bandwidths)

            # print("Spectral Centroid Mean:", spectral_centroid_mean)
            # print("Spectral Centroid Variance:", spectral_centroid_var)
            # print("Spectral Bandwidth Mean:", spectral_bandwidth_mean)
            # print("Spectral Bandwidth Variance:", spectral_bandwidth_var)

            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_mean = np.mean(rolloff)
            rolloff_var = np.var(rolloff)
            # print(rolloff_mean)
            # print(rolloff_var)


            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            zero_crossing_rate_var = np.var(zero_crossing_rate)
            # print(zero_crossing_rate_mean)
            # print(zero_crossing_rate_var)


            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmony_mean = np.mean(y_harmonic)
            harmony_var = np.var(y_harmonic)
            # print(harmony_mean)
            # print(harmony_var)


            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            perceptr_mean = np.mean(mfccs)
            perceptr_var = np.var(mfccs)
            # print(perceptr_mean)
            # print(perceptr_var)


            tempo, _ = librosa.beat.beat_track(y=y, sr = sr)

            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            # Calculate mean and variance for each MFCC
            mfcc_means = np.mean(mfccs, axis=1)  # Compute mean along each MFCC
            mfcc_vars = np.var(mfccs, axis=1)    # Compute variance along each MFCC

            # Extract individual means and variances for each MFCC
            mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean, mfcc6_mean, mfcc7_mean, mfcc8_mean, mfcc9_mean, mfcc10_mean, mfcc11_mean, mfcc12_mean, mfcc13_mean, mfcc14_mean, mfcc15_mean, mfcc16_mean, mfcc17_mean, mfcc18_mean, mfcc19_mean, mfcc20_mean = mfcc_means

            mfcc1_var, mfcc2_var, mfcc3_var, mfcc4_var, mfcc5_var, mfcc6_var, mfcc7_var, mfcc8_var, mfcc9_var, mfcc10_var, mfcc11_var, mfcc12_var, mfcc13_var, mfcc14_var, mfcc15_var, mfcc16_var, mfcc17_var, mfcc18_var, mfcc19_var, mfcc20_var = mfcc_vars


            # Example usage
            data = {
                    "id": [filename+"."+str(index/3)],
                    "chroma_stft_mean": [chroma_stft_mean],
                    "chroma_stft_var": [chroma_stft_var],
                    "rms_mean": [rms_mean],
                    "rms_var": [rms_var],
                    "spectral_centroid_mean": [spectral_centroid_mean],
                    "spectral_centroid_var": [spectral_centroid_var],
                    "spectral_bandwidth_mean": [spectral_bandwidth_mean],
                    "spectral_bandwidth_var": [spectral_bandwidth_var],
                    "rolloff_mean": [rolloff_mean],
                    "rolloff_var": [rolloff_var],
                    "zero_crossing_rate_mean": [zero_crossing_rate_mean],
                    "zero_crossing_rate_var": [zero_crossing_rate_var],
                    "harmony_mean": [harmony_mean],
                    "harmony_var": [harmony_var],
                    "perceptr_mean": [perceptr_mean],
                    "perceptr_var": [perceptr_var],
                    "tempo": [tempo],
                    "mfcc1_mean": [mfcc1_mean],
                    "mfcc1_var": [mfcc1_var],
                    "mfcc2_mean": [mfcc2_mean],
                    "mfcc2_var": [mfcc2_var],
                    "mfcc3_mean": [mfcc3_mean],
                    "mfcc3_var": [mfcc3_var],
                    "mfcc4_mean": [mfcc4_mean],
                    "mfcc4_var": [mfcc4_var],
                    "mfcc5_mean": [mfcc5_mean],
                    "mfcc5_var": [mfcc5_var],
                    "mfcc6_mean": [mfcc6_mean],
                    "mfcc6_var": [mfcc6_var],
                    "mfcc7_mean": [mfcc7_mean],
                    "mfcc7_var": [mfcc7_var],
                    "mfcc8_mean": [mfcc8_mean],
                    "mfcc8_var": [mfcc8_var],
                    "mfcc9_mean": [mfcc9_mean],
                    "mfcc9_var": [mfcc9_var],
                    "mfcc10_mean": [mfcc10_mean],
                    "mfcc10_var": [mfcc10_var],
                    "mfcc11_mean": [mfcc11_mean],
                    "mfcc11_var": [mfcc11_var],
                    "mfcc12_mean": [mfcc12_mean],
                    "mfcc12_var": [mfcc12_var],
                    "mfcc13_mean": [mfcc13_mean],
                    "mfcc13_var": [mfcc13_var],
                    "mfcc14_mean": [mfcc14_mean],
                    "mfcc14_var": [mfcc14_var],
                    "mfcc15_mean": [mfcc15_mean],
                    "mfcc15_var": [mfcc15_var],
                    "mfcc16_mean": [mfcc16_mean],
                    "mfcc16_var": [mfcc16_var],
                    "mfcc17_mean": [mfcc17_mean],
                    "mfcc17_var": [mfcc17_var],
                    "mfcc18_mean": [mfcc18_mean],
                    "mfcc18_var": [mfcc18_var],
                    "mfcc19_mean": [mfcc19_mean],
                    "mfcc19_var": [mfcc19_var],
                    "mfcc20_mean": [mfcc20_mean],
                    "mfcc20_var": [mfcc20_var]
                }

            file_save_name = '/Users/kailaiwang/Documents/CS4701/my_csv/my_csv.csv'

            # Extract the list of floats from the data dictionary
            new_data = [value[0] for value in data.values()]

            # Append the new data to the CSV file
            add_line_to_csv(file_save_name, new_data)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print("index",index)




