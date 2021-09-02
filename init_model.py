

import tflearn
from tensorflow.python.framework import ops

def init_model(X_train,y_train,X_test,y_test):
    
    ops.reset_default_graph()
    
    net = tflearn.input_data(shape=[None, len(X_train[0])])
    # 425
    net = tflearn.fully_connected(net, 550)
    net = tflearn.fully_connected(net, 550)
    
    # net = tflearn.fully_connected(net, 350)
    # net = tflearn.fully_connected(net, 64)
    net = tflearn.fully_connected(net, len(y_train[0]), activation = "softmax")
    # 141
    net = tflearn.regression(net)
    
    model = tflearn.DNN(net)
    
    import os
    
    
    if os.path.exists("model.tflearn.meta"):
        model.load("model.tflearn")
    else:
        model.fit(X_train, y_train,validation_set=(X_test,y_test), n_epoch=10, batch_size=8, show_metric=True)
        model.save("model.tflearn")
        
    return model
