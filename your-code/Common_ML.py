# define regression model
def classification_model(n_cols, n_classes=2, added_layers=0,learning_rate=0.01, custom_optimizer=False):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Input

    model = Sequential()
    model.add(Input(shape=(n_cols,)))            # input: number of features
    model.add(Dense(50, activation='relu'))      # always at least one hidden layer

    for i in range(added_layers):
        model.add(Dense(50, activation='relu'))

    model.add(Dense(n_classes, activation='softmax'))   # one unit per class

    # custom optimizer with your own learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if custom_optimizer:
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score
    import numpy as np

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))