import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


directory = '.'

# network for synthetic data
activation = tf.tanh
hidden_layer = [50, 50, 50]
n_outputs = 1
learning_rate = 1e-3
tf.get_default_graph()

with tf.name_scope('inputs_processing'):
        X_input = tf.placeholder(tf.float32, shape = (None, 3), name ='X_input') #S, K, T
        X_input_ = tf.placeholder(tf.float32, shape = (None, 2), name ='X_input_') #S_, T_
        r = tf.fill([tf.shape(X_input)[0],1], 0., name = 'r') #interest rate
        S = tf.slice(X_input, (0,0), (-1,1))
        K = tf.slice(X_input, (0,1), (-1,1))
        T = tf.slice(X_input, (0,2), (-1,1))
        X = tf.concat([S/(K*tf.exp(-r*T)), T], 1)#input matrix for ANN
        S_ = tf.slice(X_input_, (0,0), (-1,1))
        T_ = tf.slice(X_input_, (0,1), (-1,1))
        X_ = tf.concat([S_/(K*tf.exp(-r*T_)), T_], 1)#input matrix for ANN_

with tf.name_scope('ann'):
        def ann(x, hidden_layer, n_outputs, activation, reuse = False):
                Z = tf.layers.dense(x, hidden_layer[0], activation = activation,name = 'hidden1', reuse = reuse)
                for i in range(1, len(hidden_layer)):
                        Z = tf.layers.dense(Z, hidden_layer[i], activation =activation, name = 'hidden' + str(i+1), reuse = reuse)
                return tf.layers.dense(Z, n_outputs, name = 'out', reuse = reuse)
        out = ann(X, hidden_layer, n_outputs, activation) #out is ANN estimate
        out = tf.where(tf.greater(T, 1e-3), out, tf.maximum(S/K - 1, 0)) #if
        out = K*out # multiply (C/K) by K to obtain C#derivatives of option price is computed
        delta = tf.gradients(out, S)[0]
        theta = tf.gradients(out, T)[0]
        gamma = tf.gradients(delta, S)[0]#same as above, but for option price at (t+h)
        out_ = ann(X_, hidden_layer, n_outputs, activation, reuse = True)
        out_ = K*tf.where(tf.greater(T_, 1e-3), out_, tf.maximum(S_/K - 1, 0))
with tf.name_scope('loss'):
        hedging_mse = tf.losses.mean_squared_error(labels = delta*(S_-S),predictions = (out_-out)) #this is the loss (objective) function,
with tf.name_scope('training'):
        optimizer = tf.train.AdamOptimizer(learning_rate) #ADAM optimization
        training_op = optimizer.minimize(hedging_mse)
with tf.name_scope('init_and_saver'):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

# Simulating geometric Brownian motion
def stock_sim_path(S, alpha, delta, sigma, T, N, n):
    """Simulates geometric Brownian motion."""
    h = T/n
    mean = (alpha - delta - .5*sigma**2)*h
    vol = sigma * h**.5
    return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))

def get_batch2(stock_path,n, moneyness_range = (.5,2)): 
    """Constructs theoretical options based on the time series stock_path"""
    picks = np.random.randint(0, len(stock_path)-1, n)
    T = np.random.randint(1, 500, (n,1))
    S = stock_path[picks]
    S_ = stock_path[picks+1]
    K = np.random.uniform(*moneyness_range, (n,1))*S
    X = np.hstack([S, K, T/250])
    X_ = np.hstack([S_, (T-1)/250])
    return X, X_


        
# define neural network architecture
ann = tf.keras.Sequential(
    layers=
        [tf.keras.layers.Dense(hidden_layer[0], activation = activation, input_shape=(2,))] + \
        [tf.keras.layers.Dense(hidden_layer[i], activation = activation) for i in range(1, len(hidden_layer))] + \
        [tf.keras.layers.Dense(n_outputs, activation = tf.keras.activations.softplus)],
    name="ann")


#model training
#model training
n_epochs = 500 #number of training epochs
n_batches = 1000 #number of batches per epoch
batch_size = 10000 #number of theoretical options in each batch
T = 2 #years of training data
days = int(250*T)
stock_path = stock_sim_path(100, .05, 0, .15, T, 1, days) #simulate stock
stock_path_test = stock_sim_path(100, .05, 0, .15, T, 1, days) #simulate
#plot stock paths

plt.plot(stock_path, label = 'Training')
plt.plot(stock_path_test, label = 'Test')
plt.legend()
plt.show()
X_test, X_test_ = get_batch2(stock_path_test, batch_size) #get test-set
with tf.Session() as sess: #start tensorflow session
        init.run() #initialize variables
        for epoch in range(n_epochs):
                for batch in range(n_batches):
                        X_train, X_train_ = get_batch2(stock_path, batch_size) #get
                        sess.run([training_op], feed_dict = {X_input: X_train,X_input_: X_train_}) #training operation
                epoch_loss = hedging_mse.eval({X_input: X_test, X_input_:X_test_})
                print('Epoch:', epoch, 'Loss:', epoch_loss, 'BS Loss:',bs_hedging_mse.eval({X_input: X_test, X_input_: X_test_}))
        save_path = saver.save(sess, directory + '/ann_save.ckpt') #save model

