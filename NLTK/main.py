import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Corpus - A large collection of data
# .categories displays the various categories on which data is present in the brown corpus
print(brown.categories())
# We can use .sents() method to get the sentences of a category
data = brown.sents(categories='adventure')
# Here data is a list of list of words. Basically data is a list of sentences.
# And each sentence is a list of words
print(data[0])
# Joining a list of words into sentence using join method
print(' '.join(data[0]))


# Bag of words pipeline
# Get the data/corpus
# Tokenisation, Stopword Removal
# Stemming/Lemmatization
# Building a vocab
# Vectorization
# Classification


# Tokenisation and Stopward Removal
document = """It was a very pleasant day. The weather was cool and there were light showers.
I went to the market to buy some fruits."""
sentence = "Send all the 50 documents related to chapters 1, 2, 3 at something@example.com"
# sent_tokenize breaks a document of multiple sentences into a list of sentences
sents = sent_tokenize(document)
print(sents)
print(len(sents))
# word_tokenize breaks a sentence into a list of words. We could also use split method here and
# create our own custom tokenization
words = word_tokenize(sentence)
print(words)
print(len(words))
# Stopwords
sw = set(stopwords.words('english'))
print(sw)


# Creating a function to remove stopwords
def remove_stopwords(text, stopwords):
    useful_words = [w for w in text if w not in stopwords]
    return useful_words


text = "i am bothered by her very much"
text_words = word_tokenize(text)
print(text_words)
useful_text = remove_stopwords(text_words, sw)
print(useful_text)
# Tokenization Using Regular Expressions
tokenizer = RegexpTokenizer('[a-zA-Z@]+')
useful_text = tokenizer.tokenize(sentence)
print(useful_text)


# Stemming
text = """Foxes love to make jumps. The quick brown fox was seen jumping over the 
lovely dog from a 6 feet high wall"""
ps = PorterStemmer()
print(ps.stem('jumping'))
print(ps.stem('jumped'))
print(ps.stem('jumps'))
# We see that the stemmer converts all three of the above into 'jump' only


# Lemmatization
wn = WordNetLemmatizer()
print(wn.lemmatize('jumped'))
