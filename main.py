from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap

model = models.Sequential(name="Perceptron", layers=[    layers.Dense(             #a fully connected layer
          name="dense",
          input_dim=3,        #with 3 features as the input
          units=1,            #and 1 node because we want 1 output
          activation='linear' #f(x)=x
    )
])
model.summary()