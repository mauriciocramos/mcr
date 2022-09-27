import pandas as pd
from preprocessing.combine_text_columns import combine_text_columns
from sklearn.feature_extraction.text import CountVectorizer

def ngram_vocabulary_size(text_vector, ngram_range=(1,1), tokenizer=None, token_pattern='(?u)\\b\\w+\\b'):
    if isinstance(text_vector, pd.DataFrame):
        text_vector = combine_text_columns(text_vector)
    return len(CountVectorizer(tokenizer=tokenizer, token_pattern=token_pattern, ngram_range=ngram_range).fit(text_vector).get_feature_names_out())