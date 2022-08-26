from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from textblob import TextBlob
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


def keyword_extractor(doc):

    stop_words = set(stopwords.words('english'))

    total_sentences = sent_tokenize(doc)
    total_sent_len = len(total_sentences)

    total_words = (doc.split())

    total_word_length = len(doc.split())

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())

    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    print(tf_idf_score)

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    return(get_top_n(tf_idf_score, 5))

def predict_sentiment(review):
    analysisPol = TextBlob(review).polarity
    return analysisPol

def actions(keywords):
    if 'packaging' in keywords:
        return "Packaging should be improved. We suggest you to use Bubble wrap to prevent any breakage."
    if 'quality' in keywords:
        return "We received some feeback regarding the product quality. In case you lack resources to purchase good quality raw materials, you can contact our helpline 123456789"
    if 'time' in keywords:
        return "There was a delay in delivery. This might be at the shipping agency's end but we recommend you check out the product demand forecast in your dashboard to stock up in advance and serve your customers within a shorter span of time."
    if 'customisation' in keywords:
        return "Product was not received as per customisation request. Please double check"
    if 'prices' in keywords:
        return "We received a general request to lower the prices. To get to know the general trend of prices of such products, please have a look at the profiles of other similar users."
    else:
        return "Null"




def final_extraction(review):
    sentence_split = review.split(',')
    print(sentence_split)
    for sentence in sentence_split:
        sentence_lower = sentence.lower()
        sentiment = predict_sentiment(sentence_lower)
        print(sentiment)
        if (sentiment < 0):
            keywords = keyword_extractor(sentence_lower)
            return actions(keywords)


app = FastAPI()

class request_body(BaseModel):
    review : str

@app.get('/predict')
def predict(review:str):
    recommended_action = final_extraction(review)
    return { 'action' : recommended_action}



# sentence = input()
# print(final_extraction(sentence))
#


# mapped_keys = ['shipping','texture','paint','cracked'];











