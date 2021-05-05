# Political-Frame-Prediction-using-Congressional-Tweets
Congressional Tweets dataset for political framing identification

The aim of this project can be considered as a text classification task.  Firstly, we can implement a basic classifier using Bag of n-grams feature representations as the baselines. 5 different models jhave been used for the implementation of the classifiers:

  1. Support vector machine (SVM);
  2. Logistic regression with thel2regularization (LR);
  3. A  feed-forward  neural  network  with  1  hidden  layer  (NN1)  and  100  neurons  in  the  hidden layer;
  4. A feed-forward neural network with 2 hidden layers (NN2) and 100 neurons in each hiddenlayer; and
  5. The tuned BERT base model (uncased).

Some extensions of the baseline have been applied in two different ways:
  1. The use of real world knowledge such as unigram supervision, political slogan using bi-grammar or tri-grammar representation, and political party imformation;
  2. Learn Representations (Word2Vec) to improve model performances.
