# FAKE-NEWS-DETECTION 
This is basic logistic regression model to predict whether news is fake or not. 

## Overview  

- Using <code>pandas</code>, python library we will read csv(Comma Seperated Value) files and store data in <code>numpy</code> arrays. 

- As **stopwords** (E.g: i, am, they, there) doesn't affect much in predicting a news as fake or real, so we will remove stopwords from whole data to make our processing fast and efficient using <code>nltk(Natural Language Tool Kit)</code> library. 

- One more way to reduce size of data can be done using **Stemming**. Here I have used <code>PorterStemmer</code> to implement stemming. It also comes under <code>nltk(Natural Language Tool Kit)</code> library. 

- I have converted textual data to numeric data using <code>TfidfVectorizer</code> which is provided by <code>sklearn(sci-kit learn)</code> library. 
- Split data into *train* and *test* data. 

- Directly apply **Logistic Regression** function using <code>sklearn</code> library. 

- Calculate *accuracy* using function <code>accuracy_score</code>. 

## Further information 

- **Stemming** is the process to convert word into its root word. Example: actor, actress -> act.

- Using <code>PorterStemmer</code> I have removed all symbols and numbers, because main thing which make difference between real news and fake news is textual part. ALso, converted whole textual data to lowercase letters and removed stopwords.

- I have reduced number of features by merging *title* and *author* column.  

- <code>TfidfVectorizer</code> stands for *Term Frequency*(It detects the word which is repeting in one sample), *Inverse Document Frequency*(It detects the word which is gets repeated in different sample).
