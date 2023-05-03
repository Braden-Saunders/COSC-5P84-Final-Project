import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()

dataset = [post.text for post in posts]
labels = [post.get('class') for post in posts]

# 80/20 data split
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, train_size=0.8, stratify=labels, random_state=123)

vectorizer = TfidfVectorizer(analyzer='word')

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print("Training model...")
start_time = time.time()

gb = GradientBoostingClassifier(n_estimators=400, random_state=0)
gb.fit(x_train, y_train)

print("--- Finished training in %s seconds ---" % (time.time() - start_time))

predictions_rf = gb.predict(x_test)
print(classification_report(y_test, predictions_rf))
