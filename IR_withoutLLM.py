#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install PyPDF2


# In[2]:


pip install nltk


# In[5]:


pip install scikit-learn


# In[6]:


pip install whoosh


# In[8]:


pip install numpy


# In[9]:


pip install matplotlib


# In[10]:


import PyPDF2


# In[34]:


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text
pdf_text = extract_text_from_pdf('Industrial+and+Systems+Engineering+M.S.I.SY.E..pdf')
print(pdf_text)


# In[35]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download


# In[36]:


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens= [stemmer.stem(token) for token in tokens]
    return tokens

processed_text = preprocess_text(pdf_text)
print(processed_text)


# In[21]:


from collections import defaultdict

def create_inverted_index(tokens):
    index = defaultdict(list)
    for i, token in enumerate(tokens):
        index[token].append(i)
    return index

inverted_index = create_inverted_index(processed_text)
print(inverted_index)


# In[38]:


def search(query, index):
    query_token = preprocess_text(query)
    results = {}
    for token in query_token:
        if token in index:
            results[token] = index[token]
    return results

query = "What are the required courses?"
search_results = search(query, inverted_index)

print(search_results)


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer

def rank_documents(query, documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    scores = (query_vec * tfidf_matrix.T).toarray()
    ranked_indices = scores.argsort()[0][::-1]
    return ranked_indices, scores
    
documents = [pdf_text]
ranked_indices, scores = rank_documents( query, documents)

print(ranked_indices)
print(scores)


# In[ ]:




