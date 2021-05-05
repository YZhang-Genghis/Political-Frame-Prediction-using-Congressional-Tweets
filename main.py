#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import collections
import re
import math
import string
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
stop = stopwords.words('english')
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from random import randrange


# In[2]:


# external diactionary for the unigram representation

dic_unigram = {}

Economic = ["premium", "premiums", "small", "business", "businesses", "tax", "taxes", "economy", "economic", "cost", 
 "employment", "market", "spending", "billion", "billions", "million", "millions", "company", "companies", 
 "funding",  "regulation", "benefit", "benefits", "health"]

dic_unigram[0] = Economic
    
Capacity = ["resource", "resources", "housing", "infrastructure", "IRS", "national", "provide", "provides", 
            "providing", "fund", "funds", "funding", "natural", "enforcement"]

dic_unigram[1] = Capacity

Morality = ["moral", "religion", "religions", "religious", "honor", "honorable", "responsible",
            "responsibility", "illegal", "protect", "god", "gods", "sanctity", "Islam",
            "Muslim", "Christian", "radical", "violence", "victim", "victims", "church"]

dic_unigram[2] = Morality
    
Fairness = ["fair", "fairness", "equal", "equality", "inequality", "law", "laws", "right", "rights", "race",
            "gender", "class", "access", "poor", "civil", "justice", "social", "women", "womens", 
            "LGBT", "LGBTQ", "discrimination", "decision", "decisions"]

dic_unigram[3] = Fairness

Legality = [ "law", "laws", "right", "rights", "executive", "ruling", "constitution", "constitutional",
            "amnesty", "decision", "decisions", "reproductive", "legal", "legality", "court", "SCOTUS", 
            "immigration", "amendment", "amendments", "judge", "authority", "precedent", "legislation"]

dic_unigram[4] = Legality

Crime = ["crime", "crimes", "criminal", "criminals", "gun", "guns", "violate", "violates", "enforce", "enforces", 
         "enforced", "enforcement", "civil", "tribunals", "justice", "victim", "victims", "civilian", 
         "civilians", "kill", "murder", "hate", "genocide", "consequences"]

dic_unigram[5] = Crime

Security = ["security", "secure", "defense", "defend", "threat", "threats", "terror", "terrorism", 
            "terrorist", "terrorists", "gun", "guns", "attack", "attacks", "wall", "border", "safe",
            "safety", "violent", "violence", "ISIS", "ISIL", "suspect", "suspects", "domestic", 
            "prevent", "protect"]

dic_unigram[6] = Security

Health = ["health", "healthy", "care", "healthcare", "Obamacare", "access", "disease", "diseases", 
          "mental", "physical", "affordable", "coverage", "quality", "uninsured", "insured", "disaster", 
          "relief", "unsafe", "cancer", "abortion"]

dic_unigram[7] = Health

Quality = ["quality", "happy", "social", "community", "life", "benefit", "benefits", "adopt",
           "fear", "deportation", "living", "job", "jobs", "activities", "family"]

dic_unigram[8] = Quality

Culture = ["identity", "social", "value", "values", "Reagan", "Lincoln", "conservatives", "conservative", 
           "liberal", "liberals", "nation", "America", "American", "Americans", "community", "communities", 
           "country", "dreamers", "immigrants", "refugees", "history", "historical"]

dic_unigram[9] = Culture

Public = ["public", "sentiment", "opinion", "poll", "polls" , "turning", "survey", "support",  "American", "Americans", 
          "reform", "action", "want", "need", "vote"]

dic_unigram[10] = Public

Factors = ["politic", "politics", "political", "stance", "view", "partisan", "bipartisan", "filibuster", "lobby", 
           "Republican", "Republicans", "Democrat", "Democrats", "House", "Senate", "Congress", "committee", "party", 
           "POTUS", "SCOTUS", "administration", "GOP"]

dic_unigram[11] = Factors

Policy = ["policy", "fix", "fixing", "work", "works", "working", "propose", "proposed", "proposing",
          "proposal", "solution", "solve", "outcome", "outcomes", "bill", "law", "amendment", "plan", 
          "support",  "repeal", "reform"]

dic_unigram[12] = Policy

External = ["regulation", "US", "ISIS", "ISIL", "relations", "international", "national", "trade", "foreign", 
            "state", "border", "visa", "ally", "allies", "united", "refugees", "leadership", "issues", "Iraq", 
            "Iran", "Syria", "Russia", "Europe", "Mexico", "Canada"]

dic_unigram[13] = External

Factual = ["health", "insurance", "affordable", "deadline", "enroll", "sign", "signed", "program", "coverage"]

dic_unigram[14] = Factual

Promotion = ["statement", "watch", "discuss", "hearing", "today", "tonight", "live", "read", 
             "floor", "talk", "tune", "opinion", "TV", "oped"]

dic_unigram[15] = Factual

Support = ["victims", "thoughts", "prayer", "prayers", "praying", "family", "stand", "support", "tragedy", 
           "senseless", "condolences"]

dic_unigram[16] = Factual


# In[ ]:


# word2vec --- potential to use. Comment out

#from gensim.models import Word2Vec
#sentences = train_data["token"]
#model = Word2Vec(sentences, min_count=1)

# function for the maximum similarity generalization

def MaxSim(token):
    flag = False
    res  = []
    curr = -1
    ans = 0
    for w in token:
        for i in range(len(dic_unigram)):
            for key in dic_unigram[i]:
                if w in model.wv.vocab and key in model.wv.vocab:
                    maxSim = model.wv.similarity(w, key)
                    if maxSim >= curr:
                        flag = key
                        curr = maxSim
                        #print(w)
                    elif maxSim == curr:
                        res.append(flag)
                    else:
                        continue
    return flag


# In[3]:


def ReadFile(input_csv_file):
    # read the data
    train_data = pd.read_csv(input_csv_file)
    
    # lower case the text columm
    train_data["text"] = train_data["text"].str.lower()
    
    # remove punctuations
    train_data["text"] = train_data["text"].str.replace('[{}]'.format(string.punctuation), '')
    train_data["text"] = train_data["text"].str.replace(r'[^\w\s]+', '')
    train_data["text"] = train_data["text"].str.replace('\d+', '')
    train_data["text"] = train_data["text"].str.replace("rt", '')

    # remove stop words
    train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # tokenize the text column
    train_data["token"] = train_data["text"].str.split()
    
    # bag of words
    texts = train_data["text"]
    bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in texts]
    sumbags = sum(bagsofwords, collections.Counter())

    # create the bag of bi-grams
    # words =  train_data["token"]
    # bagsofbigrams = [Counter(ngrams(txt, n=2)) for txt in words]
    # sumofbigrams = sum(bagsofbigrams, collections.Counter())

    # create the bag of tri-grams
    # bagsoftrigrams = [Counter(ngrams(txt, n=3)) for txt in words]
    # sumoftrigrams = sum(bagsoftrigrams, collections.Counter())

    # creat bag of issues representation

    issues = train_data["issue"]
    bagsofissues = [collections.Counter(re.findall(r'\w+', txt)) for txt in issues]

    # the whole dictionary is represented
    sumissues = sum(bagsofissues, collections.Counter())
    N1 = len(sumissues)
    
    def vectorize(token):
        """Vectorize the given word"""
        vector = [0]*17
        for w in token:
            for i in range(len(dic_unigram)):
                if w in dic_unigram[i]:
                    vector[i] += 1
                else:
                    continue
        return vector

    #def vectorize_bobg(token):
    #    """Vectorize the given word"""
    #    vector = []
    #    pairs = list(zip(token, token[1:]))
    #    for w in sumofbigrams:
    #        vector.append(pairs.count(w))
    #    return vector

    #def vectorize_tribg(token):
    #    """Vectorize the given word"""
    #    vector = []
    #    pairs = list(zip(token, token[1:], token[2:]))
    #    for w in sumoftrigrams:
    #        vector.append(pairs.count(w))
    #    return vector

    def vectorize_bow(token):
        """Vectorize the given word"""
        vector = []
        for w in sumbags:
            vector.append(token.count(w))
        return vector

    def vectorize_issue(token):
        """Vectorize the given issue"""
        vector = []
        for w in sumissues:
            vector.append(token.count(w))
        return vector
    N = len(sumbags)

    train_data["unigram"]   = train_data["token"].apply(vectorize)
    train_data["bow"]       = train_data["token"].apply(vectorize_bow)
    #train_data["bobigram"]  = train_data["token"].apply(vectorize_bobg)
    #train_data["botrigram"]  = train_data["token"].apply(vectorize_tribg)
    train_data["Issue2Vec"] = train_data["issue"].apply(vectorize_issue)
    

    # code the political status (democrat == 1; republican = 0)
    train_data['status'] = np.where(train_data['author']=='democrat', 1, 0)
    train_data['status'] = train_data['status'].apply(lambda x: [x])

    # the whole feature representation -- replace the bow into tfidf if needed
    train_data["feature"] = train_data["bow"] + train_data['Issue2Vec'] + train_data['status'] + train_data["unigram"] 

    train_data["feature"] = train_data["feature"].apply(lambda x: np.asarray(x))
    
    # creat feature and target np array
    features = train_data["feature"].to_numpy()
    features = np.array(features.tolist())
    target   = train_data["label"].to_numpy()
    target   = np.array([i - 1 for i in target])

    return features, target, sumbags, sumissues


# In[4]:


def ReadTest(input_csv_file, sumbags, sumissues):
    # read the test the data
    test_data = pd.read_csv(input_csv_file)
    
    # lower case the text columm
    test_data["text"] = test_data["text"].str.lower()
    
    # remove punctuations
    test_data["text"] = test_data["text"].str.replace('[{}]'.format(string.punctuation), '')
    test_data["text"] = test_data["text"].str.replace(r'[^\w\s]+', '')
    test_data["text"] = test_data["text"].str.replace('\d+', '')
    test_data["text"] = test_data["text"].str.replace("rt", '')

    # remove stop words
    test_data['text']  = test_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # tokenize the text column
    test_data["token"] = test_data["text"].str.split()

    # creat bag of words representation
    texts = test_data["text"]
    
    ### define functions to represent the bag of words ###
    def vectorize(token):
        """Vectorize the given word"""
        vector = [0]*17
        for w in token:
            for i in range(len(dic_unigram)):
                if w in dic_unigram[i]:
                    vector[i] += 1
                else:
                    continue
        return vector
    
    def vectorize_bow(token):
        """Vectorize the given word"""
        vector = []
        for w in sumbags:
            vector.append(token.count(w))
        return vector

    def vectorize_issue(token):
        """Vectorize the given issue"""
        vector = []
        for w in sumissues:
            vector.append(token.count(w))
        return vector
    
    N = len(sumbags)
    
    test_data["bow"]       = test_data["token"].apply(vectorize_bow)
    test_data["unigram"]   = test_data["token"].apply(vectorize)

    # code the issue vector
    test_data["Issue2Vec"] = test_data["issue"].apply(vectorize_issue)

    # code the political status (democrat == 1; republican = 0)
    test_data['status'] = np.where(test_data['author']=='democrat', 1, 0)
    test_data['status'] = test_data['status'].apply(lambda x: [x])

    # the whole feature representation -- replace the bow into tfidf if needed
    test_data["feature"] = test_data["bow"] + test_data['Issue2Vec'] + test_data['status'] + test_data["unigram"]
    test_data["feature"] = test_data["feature"].apply(lambda x: np.asarray(x))
    
    # creat feature and target np array
    test_features = test_data["feature"].to_numpy()
    test_features = np.array(test_features.tolist())
    
    return test_features


# In[5]:


class NeuralNet(nn.Module):

    def __init__(self, inputs, hiddens, outputs):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(inputs, hiddens)
        self.linear2 = nn.Linear(hiddens, hiddens)
        self.linear3 = nn.Linear(hiddens, outputs)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs


# In[11]:


def LR():
    features, target, sumbags, sumissues = ReadFile('train.csv')
    test_features = ReadTest('test.csv', sumbags, sumissues)

    # logistic regression
    pred = LogisticRegression(penalty = "l2", max_iter = 10000, C = 1, class_weight = "balanced")
    
    # identify the maximum score 
    X_folds = np.array_split(features, 5)
    y_folds = np.array_split(target, 5)
    scores = list()
    for k in range(5):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        scores.append(pred.fit(X_train, y_train).score(X_test, y_test))
    print(scores)
    
    # for prediction of the new labels
    
    max_k = np.argmax(scores)
    X_train = list(X_folds)
    X_test = X_train.pop(max_k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(max_k)
    y_train = np.concatenate(y_train)
    LR_pred = pred.fit(X_train, y_train)
    
    prediction = LR_pred.predict(test_features)   
    prediction = [i + 1 for i in prediction]

    # Save predicted labels in 'test_lr.csv'
    test_data = pd.read_csv('test.csv')
    test_data["label"] = prediction
    test_data.to_csv ('test_proj.csv', index = False, header=True)


# In[12]:


# function used for NN and BERT

def cross_validation_split(dataset, folds=5):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy(prediction, target):
    """calculate the accuracy of the classification
    Inputs-
        prediction - A 1-D numpy array of predicted label
        target     - A 1-D numpy array of true label
        
    Returns-
        accuracy - accuracy in percentage
    """
    
    N = len(prediction)
    sum_count = 0
    for i in range(N):
        if prediction[i] == target[i]:
            sum_count += 1  
    accuracy = sum_count/N
    return accuracy


# In[13]:


def NN():
    
    features, target, sumbags, sumissues = ReadFile('train.csv')
    test_features = ReadTest('test.csv', sumbags, sumissues)
    all_data = np.concatenate((list(features), target[:,None]),axis=1)
   
    
    dataset_split = cross_validation_split(all_data, folds = 5)
    train = []
    test  = []
    for i in range(len(dataset_split)):
        curr = dataset_split.pop(0)
        test.append(np.array(curr))
        train.append(np.vstack((dataset_split)))
        dataset_split.append(curr)
        
    losses = []
    #val_losses = []
    #test_losses = []
    loss_function = nn.CrossEntropyLoss()
    model = NeuralNet(len(features[0]), 100, 17)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3)
    for name, module in model.named_children():
        print('resetting ', name)
        module.reset_parameters()
        
        
    scores = []
    for i in range(len(train)):
        features = train[i][:,:-1]
        target   = train[i][:, -1]
        for epoch in range(3):
            total_loss = torch.Tensor([0])

            context_var = autograd.Variable(torch.Tensor(list(features)))
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            log_probs = model(context_var)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            loss = loss_function(log_probs, autograd.Variable(
                torch.Tensor(target).long()))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            losses.append(total_loss)

            if epoch%200 == 0:
                print("Epoch No.", epoch)

        features_test = test[i][:,:-1]
        target_test   = test[i][:, -1]
        context_test = autograd.Variable(torch.Tensor(list(features_test)))
        log_probs = model(context_test)
        pred_label = torch.argmax(log_probs, dim = 1)
        s2 = accuracy(pred_label, target_test)
        print(s2)
        for name, module in model.named_children():
            print('resetting ', name)
            module.reset_parameters()
        scores.append(s2)


# In[14]:


import time
if __name__ == '__main__':
    start = time.time()
    LR()
    # comment out for not using NN()
    NN()
    end = time.time()
    print(end - start)


# In[ ]:





# In[23]:


### BERT Model, but comment out ####

"""import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, BertModel

# specify GPU
#device = torch.device("cuda")
train_data = pd.read_csv("train.csv")
# import BERT-base pretrained model
bert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_data.loc[:, "label"] = train_data["label"].apply(lambda x: x - 1)


train_text, temp_text, train_labels, temp_labels = train_test_split(train_data['text'], train_data['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=train_data['label'])

# we will use temp_text and temp_labels to create validation and test set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)



max_seq_len = 25
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,17)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
      
        # apply softmax activation
        x = self.softmax(x)
        return x

model = BERT_Arch(bert)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-1, weight_decay = 1e-3)
from sklearn.utils.class_weight import compute_class_weight
#compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts, dtype=torch.float)

# loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

# number of training epochs
epochs = 50


def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 50 batches. 
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            
        sent_id, mask, labels = batch
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate():
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        
    sent_id, mask, labels = batch
    # deactivate autograd
    with torch.no_grad():
      
        # model predictions
        preds = model(sent_id, mask)

        # compute the validation loss between actual and predicted values
        loss = cross_entropy(preds,labels)

        total_loss = total_loss + loss.item()

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

        # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    #
    
    print(f'Validation Loss: {valid_loss:.3f}')"""


# In[ ]:




