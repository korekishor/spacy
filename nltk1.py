text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
# Splits at space 
text.split()
text.split('. ')
import re 
sentences = re.compile('[.!?] ').split(text)

from nltk.tokenize import word_tokenize 
# word_tokenize(text)
# Output=['Founded', 'in', '2002', ',', 'SpaceX', '’', 's', 'mission', 'is', 'to', 'enable', 
#          'humans', 'to', 'become', 'a', 'spacefaring', 'civilization', 'and', 'a', 
#          'multi-planet', 'species', 'by', 'building', 'a', 'self-sustaining', 'city', 'on', 
#          'Mars', '.', 'In', '2008', ',', 'SpaceX', '’', 's', 'Falcon', '1', 'became', 
#          'the', 'first', 'privately', 'developed', 'liquid-fuel', 'launch', 'vehicle', 
#          'to', 'orbit', 'the', 'Earth', '.']


from nltk.tokenize import sent_tokenize
# sent_tokenize(text)
# Output= ['Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring 
#           civilization and a multi-planet \nspecies by building a self-sustaining city on 
#           Mars.', 
#          'In 2008, SpaceX’s Falcon 1 became the first privately developed \nliquid-fuel 
#           launch vehicle to orbit the Earth.]


from spacy.lang.en import English
nlp = English()
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
token_list
# Output = ['Founded', 'in', '2002', ',', 'SpaceX', '’s', 'mission', 'is', 'to', 'enable', 
#           'humans', 'to', 'become', 'a', 'spacefaring', 'civilization', 'and', 'a', 
#           'multi', '-', 'planet', '\n', 'species', 'by', 'building', 'a', 'self', '-', 
#           'sustaining', 'city', 'on', 'Mars', '.', 'In', '2008', ',', 'SpaceX', '’s', 
#           'Falcon', '1', 'became', 'the', 'first', 'privately', 'developed', '\n', 
#           'liquid', '-', 'fuel', 'launch', 'vehicle', 'to', 'orbit', 'the', 'Earth', '.']



# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
# nlp.add_pipe(sbd)

# doc = nlp(text)

# create list of sentence tokens
# sents_list = []
# for sent in doc.sents:
#     sents_list.append(sent.text)
# sents_list


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
  
ps = PorterStemmer()
  
# choose some words to be stemmed
words = ["program", "programs", "programmer", "programming", "programmers"]
  
for w in words:
    # print(w, " : ", ps.stem(w))
    pass

#  output=program program program program program program  ----- this splite in the base root 
# sentence = "Programmers program with programming languages"
#  output =  program program wit program language 



from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
  
# print("rocks :", lemmatizer.lemmatize("rocks"))
# print("corpora :", lemmatizer.lemmatize("corpora"))
  
# a denotes adjective in "pos"
# print("better :", lemmatizer.lemmatize("better", pos ="a"))

# output: rocks : rock
# corpora : corpus
# better : good

import nltk
# nltk.download()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sent = """This is a sample sentence,
				showing off the stop words filtration."""

stop_words = set(stopwords.words('English'))
 
word_tokens = word_tokenize(example_sent)
  
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
filtered_sentence = []

for w in word_tokens:
	if w not in stop_words:
		filtered_sentence.append(w)


# print(word_tokens)
# print(filtered_sentence)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
# pos_tag() --- give us type of world is verb,noun,adj
# // Dummy text

txt = "Sukanya, Rajib and Naba are my good friends. " \
    "Sukanya is getting married next year. " \
    "Marriage is a big step in one’s life." \
    "It is both exciting and frightening. " \
    "But friendship is a sacred bond between people." \
    "It is a special kind of love between us. " \
    "Many of you must have tried searching for a friend "\
    "but never found the right one."
 
# sent_tokenize is one of instances of use for divide sentenses
# PunktSentenceTokenizer from the nltk.tokenize.punkt module
 
tokenized = sent_tokenize(txt)
for i in tokenized:
     
    # Word tokenizers is used to find the words
    # and punctuation in a string
    wordsList = nltk.word_tokenize(i)
 
    # removing stop words from wordList
    wordsList = [w for w in wordsList if not w in stop_words]
 
    #  Using a Tagger. Which is part-of-speech
    # tagger or POS-tagger.
    tagged = nltk.pos_tag(wordsList)
 
    # print(tagged)


# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# Reads ‘alice.txt’ file
sample = open(r"C:\Users\Kishor Kore\Desktop\git_upload.txt")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
	temp = []
	
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,
							vector_size = 100, window = 5)

# Print results
print("Cosine similarity between 'alice' " +
			"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
				"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
											window = 5, sg = 1)

# Print results
print("Cosine similarity between 'alice' " +
		"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))

# from spacy.vocab import Vocab
# vocab = Vocab(strings=["hello", "world"])
# print(vocab)


 