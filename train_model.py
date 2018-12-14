import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
    '''
    Keras LSTM model require 3-Dimensional Tensors.
    Function convert samples to 3D tensors !
    '''
    X_len = len(_X)
    result_array = []
    for sample in range (0,X_len):
        sample_array = []
        for vector_idx in range (0, _vectors_per_sample):
            start = vector_idx * _vector_dim
            end = start + _vector_dim
            sample_array.append(_X[sample][start:end])
        result_array.append(sample_array)

    return np.asarray(result_array)

def convert_y_to_one_hot(_y):
    '''
    Converst y integer labels (0,1,2..) to one_hot_encoding vectors
    '''
    _y = np.asarray(_y,dtype=int)
    y_flat = [item for sublist in _y for item in sublist]
    b = np.zeros((_y.size, _y.max()+1))
    b[np.arange(_y.size),y_flat] = 1
    return b

def main():
    filename = 'pose_data2.csv'

    data = pd.read_csv(filename)
    X = data.iloc[:, 1:].values
    Y = data.iloc[:,0:1].values

    y_hot = convert_y_to_one_hot(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_hot, test_size = 0.2, random_state = 10)

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    clf = Sequential()

    clf.add(Dense(units=25, activation='relu', input_dim=50))
    clf.add(Dropout(0.1))

    clf.add(Dense(units=50, activation='relu'))
    clf.add(Dropout(0.1))

    clf.add(Dense(units=25, activation='relu'))
    clf.add(Dropout(0.1))

    clf.add(Dense(units=10, activation='relu'))
    clf.add(Dense(units=y_hot.shape[1], activation='softmax'))

    clf.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    clf.fit(X_train, Y_train, epochs=400, batch_size=50)

    score = clf.evaluate(X_test, Y_test, batch_size=50)
    print(score)
    clf.save('pose_reg_model3.h5')


if __name__ == "__main__":
    main()
    # data = pd.read_csv('pose_data.csv')
    # X = data.iloc[:, 1:].values
    # Y = data.iloc[:,0:1].values

    # x_3d = samples_to_3D_array(50, 1, X)
    # y_hot = convert_y_to_one_hot(Y)
 