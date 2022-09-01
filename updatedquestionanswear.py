import nltk
from textwrap3 import wrap
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
import traceback
import string
import pke
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
nltk.download('stopwords')

 
text = """Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] (listen); born 17 September 1950)[a] is an 
Indian politician serving as the 14th and current prime minister of India since 2014. Modi was the chief minister of Gujarat
 from 2001 to 2014 and is the Member of Parliament from Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of
  the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the first
   prime minister to have been born after India's independence in 1947 and the longest serving prime minister from outside the 
   Indian National Congress."""


summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
summary_model = summary_model.to(device)
 
 
def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")  
  text = "summarize: "+text
 
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
   
  outs = model.generate(input_ids=input_ids,attention_mask=attention_mask,early_stopping=True,num_beams=3,num_return_sequences=1,no_repeat_ngram_size=2,min_length = 75,max_length=300)
   
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
 
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()
 
  return summary
summarized_text = summarizer(text,summary_model,summary_tokenizer)
 

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)
 
        pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
     
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
         
        extractor.candidate_weighting(alpha=1.1,threshold=0.75,method='average')
        keyphrases = extractor.get_n_best(n=15)
         
        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()
    return out

def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
   
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
 
  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords[:5]

imp_keywords = get_keywords(text,summarized_text)
 
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
 
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,attention_mask=attention_mask,early_stopping=True, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, max_length=72)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]

  Question = dec[0].replace("question:","")

  Question= Question.strip()
  return Question


for wrp in wrap(summarized_text, 150):
  print (wrp)
print ()

for answer in imp_keywords:
  ques = get_question(summarized_text,answer,question_model,question_tokenizer)
  print (ques)
  print (answer.capitalize())
  print ()

