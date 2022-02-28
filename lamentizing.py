# Step 0: Import packages and stopwords
import nltk
nltk.download('omw-1.4')
import pattern
from gensim.corpora import Dictionary
from gensim import corpora
nltk.download('stopwords')
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')
stop_words = stop_words + ['com', 'edu', 'subject', 'lines', 'organization', 'would', 'article', 'could']
dataset = api.load("text8")
data = [d for d in dataset]
# Step 2: Prepare Data (Remove stopwords and lemmatize)
data_processed = []

for i, doc in enumerate(data[:100]):
    doc_out = []
    for wd in doc:
        if wd not in stop_words:  # remove stopwords
            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
            if lemmatized_word:
                doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
        else:
            continue
    data_processed.append(doc_out)


# Print a small sample
data_processed[0][:5]

# Step 3: Create the Inputs of LDA model: Dictionary and Corpus
dct = corpora.Dictionary(data_processed)
corpus = [dct.doc2bow(line) for line in data_processed]
#> ['anarchism', 'originated', 'term', 'abuse', 'first']
# Step 4: Train the LDA model
lda_model = LdaMulticore(corpus=corpus,
                         id2word=dct,
                         random_state=100,
                         num_topics=7,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)

# save the model
lda_model.save('lda_model.model')

# See the topics
lda_model.print_topics(-1)
print(lda_model.print_topics())

# [(0, '0.001*"also" + 0.000*"first" + 0.000*"state" + 0.000*"american" + 0.000*"time" + 0.000*"book" + 0.000*"year" + 0.000*"many" + 0.000*"person" + 0.000*"new"'),
#  (1, '0.001*"also" + 0.001*"state" + 0.001*"ammonia" + 0.000*"first" + 0.000*"many" + 0.000*"american" + 0.000*"war" + 0.000*"time" + 0.000*"year" + 0.000*"name"'),
#  (2, '0.005*"also" + 0.004*"american" + 0.004*"state" + 0.004*"first" + 0.003*"year" + 0.003*"many" + 0.003*"time" + 0.003*"new" + 0.003*"war" + 0.003*"person"'),
#  (3, '0.001*"atheism" + 0.001*"also" + 0.001*"first" + 0.001*"atheist" + 0.001*"american" + 0.000*"god" + 0.000*"state" + 0.000*"many" + 0.000*"new" + 0.000*"year"'),
#  (4, '0.001*"state" + 0.001*"also" + 0.001*"many" + 0.000*"world" + 0.000*"agave" + 0.000*"time" + 0.000*"new" + 0.000*"war" + 0.000*"god" + 0.000*"person"'),
#  (5, '0.001*"also" + 0.001*"abortion" + 0.001*"first" + 0.001*"american" + 0.000*"state" + 0.000*"many" + 0.000*"year" + 0.000*"time" + 0.000*"war" + 0.000*"person"'),
#  (6, '0.005*"also" + 0.004*"first" + 0.003*"time" + 0.003*"many" + 0.003*"state" + 0.003*"world" + 0.003*"american" + 0.003*"person" + 0.003*"apollo" + 0.003*"language"')]