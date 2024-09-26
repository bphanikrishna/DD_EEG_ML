def class_fun():
    ''' classification modales  '''
    # prepare models
    models = []
    from keras.models import Sequential
    from keras.layers.core import Dense
    model = Sequential()
    model.add(Dense(27, input_dim=19, init='uniform', activation='relu'))
    model.add(Dense(35, init='uniform', activation='softmax'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    models.append(('ANN', model))

 
    return models


