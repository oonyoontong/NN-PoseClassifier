import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

    #Initialse data, split into target(y) and data(x)
    data = pd.read_csv(filename)
    X = data.iloc[:, 1:].values
    Y = data.iloc[:,0:1].values

    #Convert y into one hot encoding
    y_hot = convert_y_to_one_hot(Y)

    #Split into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_hot, test_size = 0.2, random_state = 10)

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    #Initialise sequential model
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
    clf.save('pose_reg_model.h5')


if __name__ == "__main__":
    main()