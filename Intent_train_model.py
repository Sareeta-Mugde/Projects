


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import text

import string
dig = set(string.digits)
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space


df = pd.read_csv('/home/siddhesh/workarea/sareeta/Final SRC/SVM model training and testing/impressionTRAININGdata.csv')

sentences = df['text']
y = df['label']


new_sentences = []
for sentence in sentences:
    sentence = sentence.lower()
    sentence = sentence.translate(translator)
    sentence = ''.join([i for i in sentence if i not in dig])
    new_sentences.append(sentence)

sk_stop_words = text.ENGLISH_STOP_WORDS
negation_words = ['aint', 'cannot', 'cant', 'darent', 'didnt', 'doesnt', 'dont', 'hadnt', 'hardly', 'hasnt',
				  'havent', 'havnt', 'isnt', 'lack', 'lacking', 'lacks', 'neither', 'never', 'no', 'nobody',
				  'none', 'nor', 'not', 'nothing', 'nowhere', 'mightnt', 'mustnt', 'neednt', 'oughtnt', 'shant',
				  'shouldnt', 'wasnt', 'without', 'wouldnt', '*n’t']

my_stop_words = [x for x in sk_stop_words if x not in negation_words]


new_sent = []
for sentence in new_sentences:
    sentence = ' '.join([word for word in sentence.split() if word not in my_stop_words])
    new_sent.append(sentence)

vectorizer = CountVectorizer()
vectorizer.fit(new_sent)
X_train = vectorizer.transform(new_sent)

# save the model to disk
vect_filename = '/home/siddhesh/workarea/sareeta/Final SRC/SVM model training and testing/final_vectorizer_v6.sav'
joblib.dump(vectorizer, vect_filename)

classifier = SGDClassifier(loss='log')
classifier.fit(X_train, y)

# save the model to disk
model_filename = '/home/siddhesh/workarea/sareeta/Final SRC/SVM model training and testing/final_classifier_v6.sav'
joblib.dump(classifier, model_filename)


import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import text
import string


def intent_classifier(user_ip_statement):

	digits = set(string.digits)
	translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) #map punctuation to space

	sk_stop_words = text.ENGLISH_STOP_WORDS

	negation_words = ['aint', 'cannot', 'cant', 'darent', 'didnt', 'doesnt', 'dont', 'hadnt', 'hardly', 'hasnt',
					  'havent', 'havnt', 'isnt', 'lack', 'lacking', 'lacks', 'neither', 'never', 'no', 'nobody',
					  'none', 'nor', 'not', 'nothing', 'nowhere', 'mightnt', 'mustnt', 'neednt', 'oughtnt', 'shant',
					  'shouldnt', 'wasnt', 'without', 'wouldnt', '*n’t']

	my_stop_words = [x for x in sk_stop_words if x not in negation_words]

	# Loading the Count Vectorizer
	vect_filename = '/home/siddhesh/workarea/sareeta/Final SRC/SVM model training and testing/final_vectorizer_v6.sav'
	loaded_vectorizer = joblib.load(vect_filename)

	# Loading the Logistic Regression model
	model_filename = '/home/siddhesh/workarea/sareeta/Final SRC/SVM model training and testing/final_classifier_v6.sav'
	loaded_model = joblib.load(model_filename)

	# User Input Text
	input_text = user_ip_statement

	input_text = input_text.lower()
	input_text = input_text.translate(translator)
	input_text = ''.join([i for i in input_text if i not in digits])
	input_text = ' '.join([word for word in input_text.split() if word not in my_stop_words])

	print(input_text)

	# Predict Intent
	vectorized_input = loaded_vectorizer.transform([input_text])
	result = loaded_model.predict(vectorized_input)
	if result == [0]:
	    print("Predicted Outcome: Absent")
	else:
	    print("Predicted Outcome: Present")
	print("==============================================================================")

if __name__ == "__main__":
	while True:
		try:
			user_ip = input("Enter Text:")
			intent_classifier(user_ip)
			print("=============================================================")

		except Exception as e:
			print("Error: " + str(e))

