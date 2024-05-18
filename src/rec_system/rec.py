import numpy as np
import pandas as pd


def generate_similarities_recommendations(P):
    # cosine similarity of vectors x, y is
    # sum(xi*yi)/(sqrt(sum(xi*xi))*sqrt(sum(yi*yi)))
    # ranges between -1 and 1 with -1 being opposite and 1 being same,
    # 0 being orthogonal

    num_songs = len(P)
    P_dot = P @ P.T
    magnitudes = np.sqrt(np.diag(P_dot)).reshape((num_songs, 1))
    magnitude_matrix = np.reciprocal(magnitudes @ magnitudes.T)
    cosine_similarities = P_dot * magnitude_matrix
    np.fill_diagonal(cosine_similarities, -1.0)
    recomendations = np.argmax(cosine_similarities, axis=1)
    return cosine_similarities, recomendations


def load_csv_gen_recommendations(filepath):
    df = pd.read_csv(filepath + ".csv", sep=",", header=None)
    P = np.asarray(np.array(df.values[1:, 1:-1]), dtype=float)
    labels = np.array(df.iloc[1:, -1:]).flatten()
    sim, rec = generate_similarities_recommendations(P)
    acc = rec_accuracy(rec, labels)
    return sim, rec, acc


def rec_accuracy(recommendation, labels):
    count = 0
    for i in range(len(recommendation)):
        if labels[recommendation[i]] == labels[i]:
            count += 1
    return count / len(recommendation)


def predict(song, P):
    P = np.vstack([P, song])
    _, rec = generate_similarities_recommendations(P)
    return rec[-1]
