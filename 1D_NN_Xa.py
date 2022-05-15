import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#-----------------------------------------#

length = 0.2
k = 385 #Copper (W/m K)
c_p = 0
rho = 0

#-----------------------------------------#

number_layers = 6
number_neurons = 20


class PINN:
    
    def __init__(self):

        self.model = tf.keras.Sequential() # make Sequential API instead of subclassing
        self.model.add(tf.keras.layers.InputLayer(input_shape=((5,))))

        for l in number_layers:
            self.model.add(tf.keras.layers.Dense(number_neurons, activation=tf.nn.elu))


    def pde_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_input)
            tape.watch(self.t_input)
            
            T_out = self.model(input)
            dT_dx = tape.gradient(T_out, self.x_input)
        
        d2T_dx2 = tape.gradient(T_out, dT_dx)
        dT_dt = tape.gradient(T_out, self.t_input)

        error = (rho*c_p*dT_dt) - (k*d2T_dx2)
        return error

    
    







"""

        self.dense1 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense2 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense3 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense4 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense5 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense6 = keras.layers.Dense(20, activation=tf.nn.elu)

    def run(self, input):
        x = self.dense1(input)
        x1 = self.dense2(x)
        x2 = self.dense3(x1)
        x3 = self.dense4(x2)
        x4 = self.dense5(x3)
        return self.dense6(x4)

model = Conduction_PINN()
model.build((2,0))

keras.utils.plot_model(model)
"""






