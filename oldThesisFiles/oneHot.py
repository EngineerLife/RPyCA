import math, sys
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Function one hot encodes data
#   takes in column of data to encode
#   returns numpy matrix of encoded column data
def oneHot(clmn):
    # define example
#    data = ['cold', 'cold', 'hot', 4, 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
#    values = np.array(data)
    values = clmn
#    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
#    print(integer_encoded)
#    feat = max(integer_encoded)
#    print("# of features: ",feat+1)
    # binary encode
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#    print(onehot_encoded)
#    print(onehot_encoded.shape)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#    print(inverted)

    return onehot_encoded
