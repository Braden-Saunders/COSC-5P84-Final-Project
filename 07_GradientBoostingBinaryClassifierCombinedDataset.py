import json
import nltk
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

interview_sentences_to_include = 25000
random.seed(123) # Constant random seed for consistency while fine tuning

with open('news_dialogue.json', 'r') as file: # Download dataset at link here: https://github.com/zcgzcgzcg1/MediaSum
    interviews = json.load(file)

# Only download if needed
try:
  posts = nltk.corpus.nps_chat.xml_posts()
except LookupError:
  nltk.download('punkt')
  nltk.download('nps_chat')
  posts = nltk.corpus.nps_chat.xml_posts()

interviews_dialogue = [line for interview in interviews for line in interview['utt']]
random.shuffle(interviews_dialogue) # Shuffle to ensure randomness across sources and time

dataset = []
labels = []

# First add interview dataset
for line in interviews_dialogue:
  if len(dataset) > interview_sentences_to_include:
    break;
  for sentence in nltk.sent_tokenize(line): # Break each line of dialogue into individual sentences
    if len(nltk.word_tokenize(sentence)) > 2 and '-' not in sentence and '(' not in sentence: # If sentence is longer than 2 words, and doesn't contain characters which do not typically occur in natural language
      labels.append(sentence.endswith('?'))
      dataset.append(sentence if random.random() < 0.01 else sentence.replace('?', '.')) # Keep only 1% of the question marks

# Next add nps_chat dataset
for post in posts:
  labels.append(post.get('class') in ['whQuestion', 'ynQuestion'])
  dataset.append(post.text if random.random() < 0.01 else post.text.replace('?', '.')) # Keep only 1% of the question marks

# 80/20 data split
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, train_size=0.8, stratify=labels, random_state=123)

vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w\w+\b|\?', ngram_range=(1,3))

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print('Training model...')
start_time = time.time()

gb = GradientBoostingClassifier(n_estimators = 400, random_state=0)
gb.fit(x_train, y_train)

print("--- %s seconds ---" % (time.time() - start_time))

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
