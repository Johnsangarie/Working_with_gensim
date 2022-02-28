from gensim.utils import simple_preprocess
from smart_open import smart_open
from gensim.corpora import Dictionary
import nltk
from gensim import corpora

nltk.download('stopwords')  # run once
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath = path
        self.dictionary = dictionary

    def __iter__(self):
        #global mydict  # OPTIONAL, only if updating the source dictionary.
        for line in smart_open(self.filepath, encoding='latin'):
            # tokenize
            tokenized_list = simple_preprocess(line, deacc=True)

            # create bag of words
            bow = self.dictionary.doc2bow(tokenized_list, allow_update=True)

            #update the source dictionary (OPTIONAL)
            mydict.merge_with(self.dictionary)

            # lazy return the BoW
            yield bow


# Create the Dictionary
mydict = corpora.Dictionary()

# Create the Corpus
bow_corpus = BoWCorpus(r'C:\Users\sangariej\PycharmProjects\Gensim\venv\sample.txt', dictionary=mydict)  # memory friendly


for line in bow_corpus:
    print(line)    #print(line)