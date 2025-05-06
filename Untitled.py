#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[ ]:


(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
x_test


# In[ ]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


from tensorflow.keras import layers,models

model=models.Sequential([ 
                        layers.Input(shape=x_train[1].shape),
                        layers.Dense(128, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(64, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(64, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(64, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dense(1, activation='linear')
                        ])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
history=model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.2,verbose=1)



# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Training_loss',color='blue')
plt.plot(history.history['val_loss'],label='Val_loss',color='red')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()


# In[ ]:


test_mse,test_mae=model.evaluate(x_test,y_test,verbose=1)
print("test_mse=",test_mse)
print("test_mae=",test_mae)


# In[ ]:


# Predict on test data
predictions = model.predict(x_test).flatten()
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],'r--')  
plt.xlabel("Actual Prices ($1000s)")
plt.ylabel("Predicted Prices ($1000s)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


#ASSIGNMENT 2 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import tensorflow as tf 


# In[ ]:


df=pd.read_csv("IMDB Dataset.csv")
df


# In[ ]:


df['sentiment']=df['sentiment'].replace({'positive':1,'negative':0})
df


# In[ ]:


import re
def proceess(text):
    text=text.lower()
    text=re.sub(r'<.*?>','',text)
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(r'\s+',' ',text)
    return text.strip()
df['review']=df['review'].apply(proceess)


# In[ ]:


x=df['review']
y=df['sentiment']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[ ]:


print(f"shape of training data= {x_train.shape},Labels:{y_train.shape}")
print(f"shape of testing data= {x_test.shape},Labels{y_test.shape}")


# In[ ]:


tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train_seq=tokenizer.texts_to_sequences(x_train)
x_test_seq=tokenizer.texts_to_sequences(x_test)
maxf=len(tokenizer.word_index)+1
maxlen=500


# In[ ]:


x_train_pad=pad_sequences(x_train_seq,maxlen=maxlen,padding='post')
x_test_pad=pad_sequences(x_test_seq,maxlen=maxlen,padding='post')


# In[ ]:


corpus=[text.split() for text in x_train]
w2v=Word2Vec(sentences=corpus,vector_size=100,window=5,min_count=2)
embeddingm=np.zeros((maxf,100))
for word,i in tokenizer.word_index.items():
    if word in w2v.wv:
        embeddingm[i]=w2v.wv[word]


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=maxf,weights=[embeddingm], output_dim=100 ,input_length=maxlen,trainable=False),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train_pad, y_train, epochs=20, batch_size=32,
                    validation_data=(x_test_pad, y_test))


# In[ ]:


model.summary()


# In[ ]:


y_prob=model.predict(x_test_pad)
y_pred=(y_prob>0.5).astype('int32')
c_report=classification_report(y_test,y_pred)
print("\n",c_report)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:


def predict_sentiment(review_text):
    cleaned = proceess(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        print(f"\nReview: {review_text}\nPredicted Sentiment: Positive ({prediction:.2f})")
    else:
        print(f"\nReview: {review_text}\nPredicted Sentiment: Negative ({prediction:.2f})")

predict_sentiment("The plot was dull and the acting was terrible.")
predict_sentiment("An amazing movie with brilliant performances.")


# In[ ]:


#3rd Assignment


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Input,Dense,AvgPool2D,GlobalAveragePooling2D,BatchNormalization,Dropout
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:


(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()


# In[ ]:


classnames=['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']


# In[ ]:


plt.figure(figsize=(10,10))
for i in range (25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i],cmap=plt.cm.binary)
    plt.xlabel(classnames[y_train[i]])
plt.show()


# In[ ]:


#reshape the input 
x_train=x_train.reshape((x_train.shape[0],28,28,1))
x_test=x_test.reshape((x_test.shape[0],28,28,1))


# In[ ]:


#normalize x 
trainx=x_train.astype('float32')
testx=x_test.astype('float32')

trainx_norm=(trainx/255.0)
testx_norm=(testx/255.0)


# In[ ]:


#encode output 
from tensorflow.keras.utils import to_categorical
y_train_c=to_categorical(y_train)
y_test_c=to_categorical(y_test)


# In[ ]:


#model architecture 
model = tf.keras.models.Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1), name='conv-layer-1'),
        BatchNormalization(),
        AvgPool2D(pool_size=(2, 2), name='pooling-layer-1'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv-layer-2'),
        BatchNormalization(),
        AvgPool2D(pool_size=(2, 2), name='pooling-layer-2'),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv-layer-3'),
        BatchNormalization(),

        GlobalAveragePooling2D(name='pooling-layer-3'),
        Dropout(0.5),

        Dense(64, activation='relu', name='dense-layer'),
        Dropout(0.3),

        Dense(len(classnames), activation='softmax', name='output-layer')
])


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:





# In[ ]:


history=model.fit(trainx_norm,y_train_c,epochs=10,batch_size=32,verbose=1,validation_data=(testx_norm,y_test_c))


# In[ ]:


accuracy=model.evaluate(testx_norm,y_test_c)
accuracy


# In[ ]:


pd.DataFrame(history.history).plot(figsize=(10,5))
plt.title('Metrics Graph')
plt.show()


# In[ ]:


predictions=model.predict(testx_norm)
predictions=tf.argmax(predictions,axis=1)
y_test=tf.argmax(y_test_c,axis=1)
y_test=tf.Variable(y_test)

print(accuracy_score(y_test,predictions))


# In[ ]:


print("classification report",classification_report(y_test,predictions))


# In[ ]:


cm=confusion_matrix(y_test,predictions)
plt.figure(figsize=(10,10))
sns.heatmap(cm,xticklabels=classnames,fmt='d',
           yticklabels=classnames,
           annot=True)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()


# In[ ]:


#try model
import random
images=[]
labels=[]

indices = random.sample(range(len(x_test)), 10)
for i in indices:
    images.append(x_test[i])
    labels.append(y_test_c[i])
images=np.array(images)
labels=np.array(labels)

fig=plt.figure(figsize=(20,8))
row=2
col=5
x=1

for image,label in zip(images,labels):
    fig.add_subplot(row,col,x)
    predictions=model.predict(tf.expand_dims(image,axis=0))
    predictions=classnames[tf.argmax(predictions.flatten())]
    label = classnames[tf.argmax(label)]
    plt.title(f"label={label} predcition={predictions}")
    plt.imshow(image)
    x += 1

    


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# Load data
df = pd.read_csv('Google_Stock_Train.csv')  # Replace with your CSV file
df.head()
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 2. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# 3. Prepare training sequences
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  

# 4. Build RNN model
model = Sequential([
    SimpleRNN(units=50, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the model
model.fit(X, y, epochs=20, batch_size=32)

# 6. Predict future prices
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# 7. Plot results
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

plt.plot(actual_prices, label="Actual")
plt.plot(predicted_prices, label="Predicted")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[ ]:




