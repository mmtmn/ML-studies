from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap

# model = models.Sequential(name="Perceptron", layers=[    layers.Dense(             #a fully connected layer
#           name="dense",
#           input_dim=3,        #with 3 features as the input
#           units=1,            #and 1 node because we want 1 output
#           activation='linear' #f(x)=x
#     )
# ])
# model.summary()


"""
# define the function
import tensorflow as tfdef binary_step_activation(x):
    ##return 1 if x>0 else 0 
    return K.switch(x>0, tf.math.divide(x,x), tf.math.multiply(x,0))
# build the model
model = models.Sequential(name="Perceptron", layers=[
      layers.Dense(             
          name="dense",
          input_dim=3,        
          units=1,            
          activation=binary_step_activation
      )
])
"""

n_features = 10
model = models.Sequential(name="DeepNN", layers=[
    ### hidden layer 1
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features+1)/2)), 
                 activation='relu'),
    layers.Dropout(name="drop1", rate=0.2),
    
    ### hidden layer 2
    layers.Dense(name="h2", units=int(round((n_features+1)/4)), 
                 activation='relu'),
    layers.Dropout(name="drop2", rate=0.2),
    
    ### layer output
    layers.Dense(name="output", units=1, activation='sigmoid')
])
model.summary()