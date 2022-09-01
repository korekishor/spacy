# import spacy 
# names entity recorganisation
# text recorganisation tool
# mahcine translations procees all documnet is all for the nlp 
# classification and information extractions 

# nlp=spacy.load("en_core_web_sm")
# text="""text added for data information"""
# # print(nlp(text))

# with open('data/wiki_us.text','r') as f:
#     text=f.read()

# print(text)
# -------------------------------------------
# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Welcome to the Data Science Learner! . Here you will learn all things about data science , machine learning , artifical intelligence and more." )
# empty_list = []
# for token in doc:
#     empty_list.append(token.lemma_)

# final_string = ' '.join(map(str,empty_list))
# print(final_string)
# -----------------------------------------
# pip install -U spacy
# python -m spacy download en_core_web_sm


# import spacy

# # Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")

# # Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")
# doc = nlp(text)

# # doc.noun_chunks-- geting all phrases from text
# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])

# # token.lemma_ -- geting all verb from text
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# # Find named entities, phrases and concepts
# # entity.text, entity.label_ ----- as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages
# for entity in doc.ents:
#     print(entity.text, entity.label_)
 
# ----------------------
from textwrap3 import wrap

text = """Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company
Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve
system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin
rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet,
Musk put out a statement from Tesla that it was “concerned” about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and
transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, “To be clear, I strongly
believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal”.  It triggered a downward spiral for Bitcoin value but
the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising
that Dogecoin “is here to stay” and another referred to Musk's previous assertion that crypto could become the world's future currency."""

for wrp in wrap(text, 150):
  print (wrp)
print ("\n")