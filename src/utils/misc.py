def genre_to_digit(genre):
    genre_mapping = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9,
    }
    return genre_mapping.get(genre, -1)


def genres_to_digits(y):
    return list(map(genre_to_digit, [y]))
