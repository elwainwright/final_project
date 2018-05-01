import pandas
import keras
import numpy
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import jaccard_similarity_score
from keras import regularizers

"""Load data."""
train_data=pandas.read_table("comp_train.txt")
dev_data=pandas.read_table("comp_test.txt")

"""Create list of tweets, list of target emotions"""
trainf=open("comp_train.txt", "r", encoding="utf-8")
devf=open("comp_test.txt", "r", encoding="utf-8")
train_tweets=[]
trainy=[]
for line in trainf:
    data=line.strip("\n").split("\t")
    train_tweets.append(data[1])
    trainy.append(data[2:])
dev_tweets=[]
devy=[]
id_tweets=[]
for line in devf:
    data=line.strip("\n").split("\t")
    id_tweets.append(data[:2])
    dev_tweets.append(data[1])
    devy.append(data[2:])
train_tweets=train_tweets[1:]
trainy=trainy[1:]
ytrain=[]
for i in range(len(trainy)):
    ytrain.append([])
    for n in trainy[i]:
        ytrain[i].append(int(n))
ytrain=numpy.asarray(ytrain)
dev_tweets=dev_tweets[1:]
devy=devy[1:]
"""
ydev=[]
for i in range(len(devy)):
    ydev.append([])
    for n in devy[i]:
        ydev[i].append(int(n))
ydev=numpy.asarray(ydev)
"""
id_tweets=id_tweets[1:]

"Create list of list of words."
train_tokens=[]
for tweet in train_tweets:
    train_tokens.append(tweet.split())
dev_tokens=[]
for tweet in dev_tweets:
    dev_tokens.append(tweet.split())

"Create vocab dictionary."
vocab={}
for tweet in train_tokens:
    for word in tweet:
        if word not in vocab.keys():
            vocab[word]=len(vocab.keys())+1

"Creat one_hot vectors."
trainx=[]
for i in range(len(train_tokens)):
    trainx.append([])
    for word in train_tokens[i]:
        trainx[i].append(vocab[word])
devx=[]
for i in range(len(dev_tokens)):
    devx.append([])
    for word in dev_tokens[i]:
        if word in vocab.keys():
            devx[i].append(vocab[word])
        else:
            devx[i].append(0)
list_length=[len(tweet) for tweet in (trainx+devx)]
length=max(list_length)
xtrain=[]
for tweet in trainx:
    xtrain.append(tweet+([0]*(length-len(tweet))))
xdev=[]
for tweet in devx:
    xdev.append(tweet+([0]*(length-len(tweet))))
xtrain=numpy.asarray(xtrain)
xdev=numpy.asarray(xdev)

"""Create model."""
model=Sequential()
model.add(Embedding(len(vocab.keys())+1, 20))
model.add(LSTM(10, activation="sigmoid", kernel_regularizer=regularizers.l2(0.05)))
model.add(Dense(10, activation="sigmoid"))
model.add(Dropout(0.1))
model.add(Dense(11, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd")
model.fit(xtrain, ytrain)

"""Evaluate model."""
predictions=model.predict(xdev)

"""Format data for submission."""
rounded=[]
for i in range(len(predictions)):
    rounded.append([])
    for emotion in predictions[i]:
        rounded[i].append(int(round(emotion)))
rounded=numpy.asarray(rounded)
labels=["ID", "Tweet", "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
info=[]
for i in range(len(id_tweets)):
    info.append(id_tweets[i]+[str(n) for n in rounded[i]])
df=pandas.DataFrame.from_records(info, columns=labels)
df.to_csv("E-C_en_pred.txt", sep="\t", index=False)