import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
# pos tells what kind of word it is like 'a' for adjective or 'v' for verb etcetra
wn = WordNetLemmatizer()
print(wn.lemmatize('jumping', pos='v'))
print(wn.lemmatize('jumped', pos='v'))
print(wn.lemmatize('jumps'))


# Building a Vocab and Vectorization
corpus = [
    'Indian cricket team will win the world cup says captain virat kohli. World Cup will be held at Sri Lanka this year',
    'We will win next Lok Sabha elections, says confident Indian PM.',
    'The nobel laurate won the hearts of the people.',
    'The movie Raazi is an exciting Indian spy thriller based upon a real story.'
]
cv = CountVectorizer()
# Converting the corpus into vectorized form
vectorized_corpus = cv.fit_transform(corpus).toarray()
print(vectorized_corpus.shape)
print(vectorized_corpus)
# Printing the vocabulary containing mapping of each word with its index
print(cv.vocabulary_)
# This denotes the number of unique words in the vocabulary
print(len(cv.vocabulary_))
# We can also do reverse mapping wherein we convert a list of numbers into a sentence
list_of_numbers = vectorized_corpus[2]
print(cv.inverse_transform(list_of_numbers))
# The above only prints the words involved in the third sentence from the corpus which was the original
# sentence so this way we can do reverse mapping
# Also note that the words are in jumbled order and not in the same order as in third sentence in corpus
# because when we vectorize we convert the sentence into bag of words as NLTK is only concerned with the
# words occuring in the sentence and not their respective order of occurence , and this is also the reason
# of why when we do reverse mapping we get them back in jumbled manner
# Vectorization with stopword removal
tokenizer = RegexpTokenizer('[a-zA-Z@]+')


# Creating a function to tokenize and remove stopwords
def myTokenizer(sentence):
    # Tokenizing
    words = tokenizer.tokenize(sentence.lower())
    # Removing stopwords
    # Note we are using the same function for stopword removal that we already created above
    words = remove_stopwords(words, sw)
    return words


cv = CountVectorizer(tokenizer=myTokenizer)
vectorized_corpus = cv.fit_transform(corpus).toarray()
print(vectorized_corpus.shape)
print(vectorized_corpus)
print(cv.vocabulary_)
# Note that the length of our vectorized_corpus vocabulary has now reduced to 33 as compared to the 43 before
# This is because we removed the non useful words by the help of stopword removal
# This is very important while working with large data as without stopword removal our data might get too big


# Working with test_corpus
test_corpus = [
    'Indian cricket rock!'
]
# Now we are not going to fit this corpus as we already have a vocab so we are just going to tranform instead
# of the fit_transform method
print(cv.transform(test_corpus).toarray())
print(cv.transform(test_corpus).toarray().shape)
# Note that while converting we have automatically identified presence of indian and cricket as they were part
# of our vocab and are hence present in the vectorized form of test corpus


# Ways to create features
# Unigram = considering each word as a feature (we have been using unigrams in the above code)
# Bigrams = considering every two consecutive words as a feature
# Trigrams = considering every three consecutive words as a feature
# n-grams = have a range of unigram, bigram, trigram....,etc as features
# TF-IDF Normalisation
sent1 = ["this is good movie"]
sent2 = ["this is good movie but actor is not present"]
sent3 = ["this is not good movie"]

# Making a bigram
cv = CountVectorizer(ngram_range=(2, 2))
# Default value of ngram_range is (1,1) which signifies unigram . Similarly you can also make a
# trigram by passing (3,3) or n-gram by passing (n,n)
docs = [sent1[0], sent2[0]]
vectorized_docs = cv.fit_transform(docs).toarray()
print(vectorized_docs)
print(cv.vocabulary_)

# Making a n-gram considering unigram, bigram, trigram as features
cv = CountVectorizer(ngram_range=(1, 3))
docs = [sent1[0], sent2[0]]
vectorized_docs = cv.fit_transform(docs).toarray()
print(vectorized_docs)
print(cv.vocabulary_)


# TF-IDF Normalisation
sent1 = "this is good movie"
sent2 = "this was good movie"
sent3 = "this is not good movie"

corpus = [sent1, sent2, sent3]

tfidf = TfidfVectorizer()
# We can fit and transform like in above case using fit_transform method
vectorized_corpus = tfidf.fit_transform(corpus).toarray()
print(vectorized_corpus)
print(vectorized_corpus.shape)
# To get the vocabulary and its mapping with the index associated with it use .vocabulary_ method
print(tfidf.vocabulary_)
print(len(tfidf.vocabulary_))
