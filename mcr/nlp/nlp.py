import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import load
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as SKLEARN_ENGLISH_STOPWORDS

from mcr.ml.preprocessing.combine_text_columns import combine_text_columns

SKLEARN_WORD_TOKENIZER_REGEX = r'(?u)\b\w\w+\b'  # SKLEARN's default selects 2+ tokens
WORD_TOKENIZER_REGEX = r'(?u)\b\w+\b'  # select 1+ tokens
NLTK_WORD_PUNCT_TOKENIZER_REGEX = r'\w+|[^\w\s]+'
EMOJI_TOKENIZER_REGEX = r"['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"

# End of sentence pattern matches:
# optional not-word symbol, any spacing symbols, optional \r, one more \n, any spacing symbols and any non-word symbols
eos_pattern = re.compile(r'(?u)\W?\s*\r?\n+\s*' + r'\W*')
# End of sentence punctuation pattern matches:
# one or more "." or "!" or "?" ate the end of the string
eosp_pattern = re.compile(r'(?u)\.+$|!+$|\?+$')


class SentenceTokenizer:
    # Based on nltk.sent_tokenize (PunktSentenceTokenizer) exposing the realign_boundaries parameter
    # TODO: train PunktSentenceTokenizer with a large collection of plaintext in the target language

    def __init__(self, language='english', realign_boundaries=True):
        self._language = language
        self._tokenizer = load(f"tokenizers/punkt/{language}.pickle")
        self._realign_boundaries = realign_boundaries

    def tokenize(self, text):
        # replacing new lines with periods before tokenization
        # text = eos_pattern.sub('. ', text)
        sentences = self._tokenizer.tokenize(text, self._realign_boundaries)
        # Removing ending punctuations (.!?) after tokenization
        # sentences = [eosp_pattern.sub('', sentence) for sentence in sentences]]
        return sentences


class WordTokenizer:
    r"""
    A tokenizer with configurable word tokenizer, language and sentence tokenization behavior.

    WordTokenizer() same as WordTokenizer(tokenizer=RegexpTokenizer(WORD_TOKENIZER_REGEX))
    WordTokenizer(preserve_line=True) does not tokenize sentences before tokenize words
    WordTokenizer(tokenizer=NLTKWordTokenizer()) reproduces nltk's recommended word_tokenize()
    WordTokenizer(tokenizer=WordPunctTokenizer()).tokenize(text)
    WordTokenizer(WhitespaceTokenizer()).tokenize(text)
    WordTokenizer(BlanklineTokenizer()).tokenize(text)
    """

    def __init__(self,
                 language='english',
                 tokenizer=None,
                 pattern=WORD_TOKENIZER_REGEX,
                 preserve_line=False,
                 realign_boundaries=True):
        r"""
        :type tokenizer: A class with a tokenize method
        :param tokenizer: instantiated nltk tokenizer object.
        :type language: str
        :param language: NLTK language string for a trained PunktSentenceTokenizer, default 'english'.
        :type preserve_line: bool
        :param preserve_line: A flag to decide whether to sentence tokenize the text or not.
        :type realign_boundaries: bool
        :param realign_boundaries: Include additional punctuation following sentences, default True.
        """
        self._tokenizer = tokenizer if tokenizer is not None else RegexpTokenizer(pattern=pattern)
        self._pattern = pattern
        self._language = language
        self._preserve_line = preserve_line
        self._realign_boundaries = realign_boundaries

    def tokenize(self, text):
        if self._preserve_line:
            sentences = [text]
        else:
            sentences = SentenceTokenizer(language=self._language,
                                          realign_boundaries=self._realign_boundaries).tokenize(text)
        return [token for sentence in sentences for token in self._tokenizer.tokenize(sentence)]

    @property
    def language(self):
        return self._language


def sentence_count(text, language='english'):
    return len(SentenceTokenizer(language=language).tokenize(text))


def word_count(text, language='english'):
    return len(WordTokenizer(language=language).tokenize(text))


def ngram_vocabulary_size(text_vector, ngram_range=(1, 1), tokenizer=WordTokenizer(language='english'),
                          token_pattern=WORD_TOKENIZER_REGEX):
    if isinstance(text_vector, pd.DataFrame):
        text_vector = combine_text_columns(text_vector)
    vec = CountVectorizer(tokenizer=tokenizer.tokenize, token_pattern=token_pattern if tokenizer is None else None,
                          ngram_range=ngram_range)
    vec.fit(text_vector)
    return len(vec.get_feature_names_out())


def statistics(documents, language='english'):
    r"""
    Example:
        statistics(s).to_frame().T\
            .style.format('{:,.0f}').format('{:.2f}', subset=['fill %', 'unique %'])\
            .background_gradient(axis=0, cmap='RdYlGn')  # subset=[...]

    :param documents: List or Series of texts
    :param language: language to be used by the tokenizer, default 'english'
    :return: Series containing NLP statistics
    """
    if isinstance(documents, str):
        documents = [documents]
    if isinstance(documents, list):
        documents = pd.Series(documents)
    rows = documents.shape[0]
    lowercase = True  # always lower case?
    documents = documents.str.lower()
    documents = documents.dropna()
    unique_documents = documents.nunique()
    dcount = documents.shape[0]
    nchars = documents.apply(len).sum()
    sentence_tokenizer = SentenceTokenizer(language=language)
    word_tokenizer = WordTokenizer(language=language)
    scount = documents.apply(lambda x: len(sentence_tokenizer.tokenize(x))).sum()
    wcount = documents.apply(lambda x: len(word_tokenizer.tokenize(x))).sum()
    unique_words = len(token_count(documents, tokenizer=word_tokenizer, lowercase=lowercase))
    data = [rows, dcount, dcount/rows*100, unique_documents, unique_documents/dcount*100, scount, scount/dcount,
            wcount, wcount/dcount, wcount/scount, unique_words, nchars, nchars/dcount, nchars/scount]
    columns = ['rows', 'documents', 'fill %', 'unique documents', 'unique %', 'sentences', 'sentences / document',
               'words', 'words / document', 'words / sentence', 'unique words', 'characters', 'characters / document',
               'chars / sentence']
    return pd.Series(data, index=columns, name=documents.name)


def token_count(text, vectorizer=CountVectorizer, tokenizer=WordTokenizer(language='english'),
                token_pattern=WORD_TOKENIZER_REGEX, lowercase=True, stop_words=None, ngram_range=(1, 1),
                min_df=1, max_df=1.0, max_features=None, vocabulary=None, dtype=None):
    """
    Usages:
        tokenizer_count(text) # default WordTokenizer(language='english')
        tokenizer_count(text, tokenizer=None)  # no sentence with 1+ chars, based on sklearn
        tokenizer_count(text, tokenizer=None, token_pattern=SKLEARN_WORD_TOKENIZER_REGEX) # no sentence, sklearn default
        tokenizer_count(text, tokenizer=NLTKWordTokenizer(), token_pattern=None)  # no sentence, nltk's word tokenizer
        tokenizer_count(text, tokenizer=WordPunctTokenizer(), token_pattern=None) # nltk WordPunctTokenizer
        tokenizer_count(text, tokenizer=WhitespaceTokenizer(), token_pattern=None) # nltk WhitespaceTokenizer
        tokenizer_count(text, tokenizer=BlanklineTokenizer(), token_pattern=None) # nltk BlanklineTokenizer
        # nltk regexp and gaps parameter: True to find separators between tokens; False to find the tokens themselves.
        tokenizer_count(text, tokenizer=RegexpTokenizer(regexp, gaps=False), token_pattern=None)
        tokenizer_count(text, tokenizer=custom_sentence_tokenizer, token_pattern=None)  # just sentence
    """

    if isinstance(text, pd.Series):
        text = text.dropna().to_list()
    if not isinstance(text, list):
        text = [text]
    name = re.findall('[A-Z][^A-Z]*', vectorizer.__name__)[0].lower()
    if dtype is None:
        dtype = np.int64 if name == 'count' else np.float64
    ngram_vectorizer = \
        vectorizer(tokenizer=tokenizer if tokenizer is None else tokenizer.tokenize,
                   token_pattern=token_pattern if tokenizer is None else None,
                   lowercase=lowercase, stop_words=stop_words, ngram_range=ngram_range, min_df=min_df, max_df=max_df,
                   max_features=max_features, vocabulary=vocabulary, dtype=dtype)
    transformed_data = ngram_vectorizer.fit_transform(text)
    return dict(sorted(dict(zip(ngram_vectorizer.get_feature_names_out(), transformed_data.sum(axis=0).A1)).items(),
                       key=lambda item: item[1], reverse=True))


def ngram_builder(corpus, vectorizer=CountVectorizer, tokenizer=WordTokenizer(language='english'), lowercase=True,
                  stop_words=None, ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=None, vocabulary=None,
                  dtype=None):
    if isinstance(corpus, pd.Series):
        corpus = corpus.dropna()
    name = re.findall('[A-Z][^A-Z]*', vectorizer.__name__)[0].lower()
    if dtype is None:
        dtype = np.int64 if name == 'count' else np.float64
    ngram_df = \
        pd.concat(
            (pd.Series(
                token_count(corpus, vectorizer=vectorizer, tokenizer=tokenizer, lowercase=lowercase,
                            stop_words=stop_words, ngram_range=(ngram, ngram), min_df=min_df, max_df=max_df,
                            max_features=max_features, vocabulary=vocabulary, dtype=dtype),
                name=name).to_frame().assign(n=np.uint8(ngram))
             for ngram in range(ngram_range[0], ngram_range[1] + 1))
        ).sort_values([name, 'n'], ascending=[False, True])
    return ngram_df.rename_axis('ngram')


def ngram_plot(ngrams, suptitle=None, rows_per_page=50, figsize=None):
    min_gram, max_ngram = ngrams['n'].agg(['min', 'max']).tolist()
    max_rows = ngrams.groupby('n').size().max()
    if figsize is None:
        figsize = (19.2 * ((max_ngram + 1) / 2), 10.8 * (max(max_rows, rows_per_page/2) / rows_per_page))
    fig, axes = plt.subplots(1, max_ngram - min_gram + 1, figsize=figsize)
    i = 0
    for n, df in ngrams.groupby('n'):
        ax = axes[i] if isinstance(axes, np.ndarray) else axes
        # TODO: replace pandas.plot by matplotlib.pyplot
        df.iloc[:, 0].plot(kind='barh', ax=ax, title=f'{n}-gram', width=0.90)
        ax.set_ylabel(None)
        if df.iloc[:, 0].dtype == 'float':
            labels = df.iloc[:, 0].apply(lambda x: f'{x:,.2f}')
        else:
            labels = df.iloc[:, 0].apply(lambda x: f'{x:,}')
        ax.bar_label(ax.containers[0], labels=labels, label_type='center', color='black')
        ax.legend(loc='right')
        ax.invert_yaxis()
        i += 1
    fig.suptitle(suptitle, y=1)
    plt.tight_layout()


def compare_dicts(a, b, search=''):
    a_minus_b = set([(k, v) for k, v in a.items() if search in k]) - set([(k, v) for k, v in b.items() if search in k])
    b_minus_a = set([(k, v) for k, v in b.items() if search in k]) - set([(k, v) for k, v in a.items() if search in k])
    if (not len(a_minus_b)) | (not len(b_minus_a)):
        print(True)
        return
    print(False)
    return a_minus_b, b_minus_a


def tokenized_stopwords(words=None, language='english', tokenizer=WordTokenizer):
    if words is None:
        if language == 'english':
            words = list(set(stopwords.words(language)) | set(SKLEARN_ENGLISH_STOPWORDS))
        else:
            words = list(set(stopwords.words(language)))
    words = (tokenizer(language=language).tokenize(word) for word in words)
    return list(set([word for new_stopwords in words for word in new_stopwords]))


def sentiments(s):
    # Returns a dataframe with sentiment metrics for a given series
    sia = SentimentIntensityAnalyzer()
    columns = ['negative sentiment', 'neutral sentiment', 'positive sentiment', 'compound sentiment']
    df = pd.DataFrame(columns=columns, index=s.index)
    df[columns] = s\
        .replace(r'^\s*$', np.nan, regex=True)\
        .apply(lambda x: [np.nan] * 4 if x is np.nan else list(sia.polarity_scores(x).values()))\
        .tolist()
    return df


def frequency_distribution(corpus, lowercase=True, tokenizer=None):
    # under development: simpler alternative to tokenizer_count, slightly slower
    fdist = FreqDist()
    if lowercase:
        corpus = [document.lower() for document in corpus]
    for document in corpus:
        for token in tokenizer(document):
            fdist[token] += 1
    return dict(sorted(dict(fdist).items(), key=lambda item: item[1], reverse=True))


# def frequency_distributions(corpus):
#     # potential function that returns sentence and word FreqDist() from the same loop
#     sentence_frequency_distribution = FreqDist()
#     word_frequency_distribution = FreqDist()
#     for document in corpus:
#         for sentence in sentence_tokenizer(document):
#             sentence_frequency_distribution[sentence.lower()] += 1
#             for word in word_tokenizer(sentence):
#                 word_frequency_distribution[word.lower()] += 1
#     return dict(sorted(dict(sentence_frequency_distribution).items(), key=lambda item: item[1], reverse=True)),\
#            dict(sorted(dict(word_frequency_distribution).items(), key=lambda item: item[1], reverse=True))
