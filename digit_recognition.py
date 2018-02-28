import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

X = train.drop(columns=['label'])
X = X.values / 255.
X = X.reshape(-1, 28, 28, 1)
y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y)


def model_func(n1=30, s1=4, p1=2, n2=40, s2=3, p2=2, n3=30, n4=10):
    model = Sequential()
    model.add(Conv2D(n1, (s1, s1), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D((p1, p1)))
    model.add(Conv2D(n2, (s2, s2), activation='relu'))
    model.add(MaxPool2D((p2, p2)))
    model.add(Flatten())
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )

    return model


model = KerasClassifier(build_fn=model_func, epochs=15, verbose=1)

pipe = Pipeline([
    ('model', model)
])

params = {
        'model__epochs': [12, 15, 18],
        'model__n1': [30, 40],
        'model__s1': [4,5,6],
        'model__p1': [1,2,3],
        'model__n2': [40,60],
        'model__s2': [3,4,5],
        'model__p2': [1,2,3],
        'model__n3': [15,30]
}

gs = GridSearchCV(pipe, param_grid=params)
gs.fit(X_train, y_train)

print('Train Score: ', gs.best_score_)
print('Best Param Dict: ', gs.best_params_)
print('Test Score: ', gs.score(X_test, y_test))

X_pred = test
X_pred = X_pred.values / 255.
X_pred = X_pred.reshape(-1, 28, 28, 1)

Y_pred = pd.DataFrame(gs.predict(X_pred), columns=['Label'])
Y_pred.index += 1

Y_pred['Label'].to_csv('submission.csv', index=True, index_label='ImageId', header=True)