import shutil #for printing purposes
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from whooshIndex import getIndex
from whoosh import scoring #set up the scoring system
from gensim.models import KeyedVectors
from sentimentAnalysis import getSentimentIndex
import pickle

indexPath = "index"
sentimentIndexPath = "sentiment"
CSVdataPath = "CSVdata"

word2VecModelPath = "word2VecModel.txt"
try:
    word2VecModel = KeyedVectors.load_word2vec_format(word2VecModelPath)
except Exception:
    word2VecModel = None

try:
    with open("Word2VecDocumentMapping.bin", "rb") as file:
            document_mapping = pickle.load(file)
except Exception:
      pass

terminal_width = shutil.get_terminal_size().columns
str_separator = "-" * terminal_width

fieldList = ["reviewDate", "authorName", "vehicleName", "reviewTitle", "reviewText", "reviewRating"]
sentimentFieldList = ["reviewDate", "authorName", "vehicleName", "reviewTitle", "reviewText", "reviewRating", "sentimentScore"]

modes = {1: 'Full text search', 2:'Sentiment analysis', 3:'Word2Vec'}
mode = 0  #chosen search engine mode
mode_min = 1
mode_max = 3
limit = 5 #max amount of documents to show

words_to_remove = set(["car", "cars", "vehicle", "vehicles", "search", "find", "named"])
stop_words = set(stopwords.words('english')).union(words_to_remove)
review_words = ["review", "reviews", "opinion", "opinions", "reviewed", "rating"]


custom_token_pattern = r'\b\w+(?::\w+)?\b'
custom_tokenizer = RegexpTokenizer(custom_token_pattern)

index = getIndex(indexPath) #retrieve the index
sentimentIndex = getSentimentIndex(sentimentIndexPath)

searcher = index.searcher(weighting=scoring.TF_IDF()) #set the scoring system



