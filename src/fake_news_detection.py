#Importing Libraries 
import numpy as np 
import pandas as pd 
import re #Regular Expression
from nltk.corpus import stopwords #Natural Language Tool Kit
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer #Term frequency inverse document frequency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score  

#Downloading stopwords
import nltk 
nltk.download('stopwords') 


#Loading dataset into pandas DataFrame 
news_dataset = pd.read_csv("train.csv")  

#counting number of missing values in the dataset 
news_dataset.isnull().sum()

#Replacing the null values in the dataset with empty string
news_dataset = news_dataset.fillna('')

#merging the author name and news title 
news_dataset['content'] = news_dataset['author'] + " " +  news_dataset['title'] 

#seperating the data & label  
X = news_dataset.drop(columns='label',axis=1)
Y = news_dataset['label']  

#We will use in-built PortStemmer function to implement stemming
port_stem = PorterStemmer()  

#function to implement stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) #Working with alphabet characters
    stemmed_content = stemmed_content.lower() #COnverting to lowercase
    stemmed_content = stemmed_content.split() #make list which contains root words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] #removing stopwords
    stemmed_content = ' '.join(stemmed_content) #joining text back out of list
    return stemmed_content 


#Applying stemming function to content
news_dataset['content'] = news_dataset['content'].apply(stemming) 

#Seperating data and content
X = news_dataset['content'].values 
Y = news_dataset['label'].values 

#Converting textual to numeric data 
#Tf-counts the number of times a significant word occured 
#idf-It detects the word which is repeatedly coming in each sample
vectorizer = TfidfVectorizer()  
vectorizer.fit(X) 
X = vectorizer.transform(X) 


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)  

model = LogisticRegression() 

model.fit(X_train,Y_train) 

#accuracy score on training data
X_train_prediction = model.predict(X_train) 
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)  

print("Accuracy score of the trainig data: ",training_data_accuracy)  

#accuracy score on testing data
X_test_prediction = model.predict(X_test) 
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)  

print("Accuracy score of the testing data: ",testing_data_accuracy) 


X_new = X_test[1] 

prediction = model.predict(X_new) 

if(prediction==0):
    print("Prediction: News is real") 
else:
    print("Prediction: News is fake")  


if(not(Y_test[1])):
    print("Actual answer: News is real") 
else: 
    print("Actual answer: News is fake") 