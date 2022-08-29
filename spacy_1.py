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
import re
re.findall()