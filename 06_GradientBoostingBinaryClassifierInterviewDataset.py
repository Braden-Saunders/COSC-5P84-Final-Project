import json
import nltk
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

interview_sentences_to_include = 100000
random.seed(123) # Constant random seed for consistency while fine tuning

with open('news_dialogue.json', 'r') as file: # Download dataset at link here: https://github.com/zcgzcgzcg1/MediaSum
    interviews = json.load(file)

interviews_dialogue = [line for interview in interviews for line in interview['utt']]

def generate_binary_labels(sentence):
    return sentence.endswith('?')

dataset = [sentence for line in interviews_dialogue[:interview_sentences_to_include] for sentence in nltk.sent_tokenize(line)] # Limit number of lines of dialogue to avoid running out of memory
labels = [generate_binary_labels(sentence) for sentence in dataset]

# 80/20 data split
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, train_size=0.8, stratify=labels, random_state=123)

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3))

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print('Training model...')
start_time = time.time()

gb = GradientBoostingClassifier(n_estimators=400, random_state=0)
gb.fit(x_train, y_train)

print("--- Finished training in %s seconds ---" % (time.time() - start_time))

predictions_rf = gb.predict(x_test)
print(classification_report(y_test, predictions_rf))

print('Test Sentence Results:')
print(gb.predict(vectorizer.transform(['if I see a bird what do I do'])))
print(gb.predict(vectorizer.transform(['what do I do if I see a bird'])))
print(gb.predict(vectorizer.transform(['what is a bird'])))
print(gb.predict(vectorizer.transform(['where do I go to see birds'])))
print(gb.predict(vectorizer.transform(['how are you'])))
print(gb.predict(vectorizer.transform(['did we win'])))
print(gb.predict(vectorizer.transform(['did we win?'])))
print(gb.predict(vectorizer.transform(['did we win the game'])))
print(gb.predict(vectorizer.transform(['did we win the game?'])))
print(gb.predict(vectorizer.transform(['anyone know how to do that'])))
print(gb.predict(vectorizer.transform(['does anyone know how to do that'])))
print(gb.predict(vectorizer.transform(['will someone please accompany me to the store'])))
print(gb.predict(vectorizer.transform(['will someone please accompany me to the store?'])))
print(gb.predict(vectorizer.transform(['this is not a question?'])))
print(gb.predict(vectorizer.transform(['words paper computer mug?'])))
print(gb.predict(vectorizer.transform(['lamp window wall desk?'])))
