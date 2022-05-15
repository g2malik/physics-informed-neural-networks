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


class PINN(keras.Model):
    
    def __init__(self, input):
        super().__init__()
        self.input = input

        self.dense1 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense2 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense3 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense4 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.dense5 = keras.layers.Dense(20, activation=tf.nn.elu)
        self.output = keras.layers.Dense(1)

    def call(self):
        x = self.dense1(self.input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.output(x)


    def pde_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            x_input = tf.convert_to_tensor(self.input[0])
            t_input = tf.convert_to_tensor(self.input[1])

            tape.watch(x_input)
            tape.watch(t_input)

            input_again = tf.stack([x_input[:, 0], t_input[:, 0]], axis=1) #copied so need to adapt
            
            T_out = self.call(input_again)
            dT_dx = tape.gradient(T_out, x_input)
        
        d2T_dx2 = tape.gradient(T_out, dT_dx)
        dT_dt = tape.gradient(T_out, t_input)

        error = (rho*c_p*dT_dt) - (k*d2T_dx2)
        del tape
        return tf.reduce_mean(tf.square(error)) #Might have to square and find average based on what the input is

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.pde_loss(training=True)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)
    
    
    def train(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)
        loss_value, grads = self.grad()

        
        return loss_value, grads

conduction_model = PINN()

