
# coding: utf-8

# In[1]:


import keras
import numpy as np


# # AND Gate

# # Build Model

# In[2]:


model_and = keras.Sequential()
model_and.add(keras.layers.Dense(units=2, input_shape=[2]))
model_and.add(keras.layers.Dense(units=1))


# In[3]:


model_and.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])


# In[4]:


xs_and = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype = float)


# In[5]:


ys_and = np.array([[0.0], [0.0], [0.0], [1.0]], dtype = float)


# In[6]:


model_and.fit(xs_and, ys_and, epochs=300, verbose=0)


# In[7]:


def prediction_and(pred):
    x = model_and.predict(pred)
    print(x)
    if x > 0.5:
        print("It's True\n")
    else:
        print("It's False\n")


# In[8]:


pred_zero_zero = np.array([[0.0,0.0]]) # Expecting False
pred_zero_one = np.array([[0.0,1.0]])
pred_one_zero = np.array([[1.0,0.0]])
pred_one_one = np.array([[1.0,1.0]])
predictions = [pred_zero_zero, pred_zero_one, pred_one_zero, pred_one_one]


# In[9]:


for pred in predictions:
    prediction_and(pred)


# In[10]:


model_and.summary()


# In[11]:


first_layer_weights_and = model_and.layers[0].get_weights()[0]
first_layer_biases_and  = model_and.layers[0].get_weights()[1]
print(first_layer_weights_and)
print(first_layer_biases_and)


# In[12]:


second_layer_weights_and = model_and.layers[1].get_weights()[0]
second_layer_biases_and  = model_and.layers[1].get_weights()[1]
print(second_layer_weights_and)
print(second_layer_biases_and)


# # OR Gate

# In[13]:


model_or = keras.Sequential()
model_or.add(keras.layers.Dense(units=2, input_shape=[2]))
model_or.add(keras.layers.Dense(units=1))


# In[14]:


model_or.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])


# In[15]:


xs_or = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype = float)


# In[16]:


ys_or = np.array([[0.0], [1.0], [1.0], [1.0]], dtype = float)


# In[17]:


model_or.fit(xs_or, ys_or, epochs=300, verbose=0)


# In[18]:


pred_zero_zero = np.array([[0.0,0.0]]) # Expecting False
pred_zero_one = np.array([[0.0,1.0]])
pred_one_zero = np.array([[1.0,0.0]])
pred_one_one = np.array([[1.0,1.0]])
predictions = [pred_zero_zero, pred_zero_one, pred_one_zero, pred_one_one]


# In[19]:


def prediction_or(pred):
    x = model_or.predict(pred)
    print(x)
    if x > 0.5:
        print("It's True\n")
    else:
        print("It's False\n")


# In[20]:


for pred in predictions:
    prediction_or(pred)


# In[21]:


model_or.summary()


# In[22]:


first_layer_weights_or = model_or.layers[0].get_weights()[0]
first_layer_biases_or  = model_or.layers[0].get_weights()[1]
print(first_layer_weights_or)
print(first_layer_biases_or)


# In[23]:


second_layer_weights_or = model_or.layers[1].get_weights()[0]
second_layer_biases_or  = model_or.layers[1].get_weights()[1]
print(second_layer_weights_or)
print(second_layer_biases_or)


# # XOR Gate

# # Create Deeper Model

# In[24]:


model_xor = keras.Sequential()
model_xor.add(keras.layers.Dense(units=16, input_shape=[2],activation="relu"))
model_xor.add(keras.layers.Dense(units=1))


# In[25]:


model_xor.compile(optimizer="sgd", loss="mean_squared_error")


# In[26]:


xs_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype = float)


# In[27]:


ys_xor = np.array([[0.0], [1.0], [1.0], [0.0]], dtype = float)


# In[28]:


model_xor.fit(xs_xor, ys_xor, epochs=300, verbose=0)


# In[29]:


pred_zero_zero = np.array([[0.0,0.0]]) # Expecting False
pred_zero_one = np.array([[0.0,1.0]])
pred_one_zero = np.array([[1.0,0.0]])
pred_one_one = np.array([[1.0,1.0]])
predictions = [pred_zero_zero, pred_zero_one, pred_one_zero, pred_one_one]


# In[30]:


def prediction_xor(pred):
    x = model_xor.predict(pred)
    print(x)
    if x > 0.5:
        print("It's True\n")
    else:
        print("It's False\n")


# In[31]:


for pred in predictions:
    prediction_xor(pred)


# In[32]:


model_xor.summary()


# In[33]:


first_layer_weights_xor = model_xor.layers[0].get_weights()[0]
first_layer_biases_xor  = model_xor.layers[0].get_weights()[1]
print(first_layer_weights_xor)
print(first_layer_biases_xor)


# In[34]:


second_layer_weights_xor = model_xor.layers[1].get_weights()[0]
second_layer_biases_xor  = model_xor.layers[1].get_weights()[1]
print(second_layer_weights_xor)
print(second_layer_biases_xor)

