import nltk

nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def generate_binary_labels(clazz):
    return clazz in ['whQuestion', 'ynQuestion']

featuresets = [(dialogue_act_features(post.text), generate_binary_labels(post.get('class'))) for post in posts]

size = int(len(featuresets) * 0.2)
train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
