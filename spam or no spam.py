import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import pickle
from keras import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

##choose GloVe vector size 
VECTOR_SIZE = 300
## Choose Max words in each email.
EMAIL_MAX_WORDS = 100   


def create_embedding_dict():
    embeddings_dict=dict()
    with open("glove.6B.{}d.txt".format(VECTOR_SIZE), 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            """ #vector = np.asarray(values[1:], "float32")
            vector = word """
            embeddings_dict[word] = [float(x) for x in values[1:]]

    glove_emails=[]
 
    data = pd.read_csv("spam_or_not_spam.csv")
    data.fillna('',inplace=True)
    emails = data["email"].tolist()

    for i in range(len(emails)):
        
        new_email = []
        for word in emails[i].split(" "):
            if word in embeddings_dict.keys():
                new_email.append(embeddings_dict[word])
        
        if len(new_email)>EMAIL_MAX_WORDS:
            new_email = new_email[0:EMAIL_MAX_WORDS]
        else:
            padSize = EMAIL_MAX_WORDS - len(new_email)
            padding = [[0]*VECTOR_SIZE]*padSize
            new_email+=padding
        glove_emails.append(new_email)    

    with open('email_vectors_es{}_vs{}.pickle'.format(EMAIL_MAX_WORDS,VECTOR_SIZE),'wb') as handle:
        pickle.dump(glove_emails,handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_model():
    model = Sequential()
    model.add(LSTM(int(EMAIL_MAX_WORDS/2), input_shape=(EMAIL_MAX_WORDS, VECTOR_SIZE)))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation = "sigmoid"))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


def rate_model(ytest,ypred):
    f1 = f1_score(ytest,ypred,average='weighted')
    precision = precision_score(ytest, ypred, average='weighted')
    recall = recall_score(ytest,ypred,average='weighted')

    print("F1 score: ",int(f1*100)/100)
    print("Precision score: ",int(precision*100)/100)
    print("Recall score: ",int(recall*100)/100)





if __name__=="__main__":
    labels = pd.read_csv("spam_or_not_spam.csv")["label"].tolist()
   
    
    try:
        with open('email_vectors_es{}_vs{}.pickle'.format(EMAIL_MAX_WORDS,VECTOR_SIZE), 'rb') as handle:
            emails = pickle.load(handle)
            print("emails loaded !")
    
    except:
        create_embedding_dict() # requires the downlaod of GloVe's word embeddings txt files.

    Xtrain,Xtest,Ytrain,Ytest = train_test_split(emails, labels, test_size=0.2)
    
    model = create_model()
    model.fit(Xtrain,Ytrain,batch_size=64 ,epochs=40,verbose=1)
    
    ypreds = model.predict(Xtest)
    
    ypreds = [int(x>=0.5) for x in ypreds]
    rate_model(Ytest,ypreds)
   


    