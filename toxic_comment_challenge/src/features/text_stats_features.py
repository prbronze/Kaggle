import numpy as np
import pandas as pd

def text_stats_features(data_set):
    """
    Create new features from text in 'comment_text' columns from the dataframe
    provided and return new dataframe.
    
    Parameters
    ----------
    data_set : DataFrame
        DataFrame with column 'comment_text' from which to extract stats features.
   
    Returns
    -------
    df : DataFrame
        DataFrame passed to function with extracted features added as columns to it.
    """
    
    df = data_set.copy()
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text']\
        .apply(lambda comment: sum(1 for c in comment if c.isupper()))
    
    df['caps_vs_length'] = np.round(df\
        .apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1),4)
    
    df['num_exclamation_marks'] = df['comment_text']\
        .apply(lambda comment: comment.count('!'))
    
    df['num_question_marks'] = df['comment_text']\
         .apply(lambda comment: comment.count('?'))
    
    df['num_punctuation'] = df['comment_text']\
         .apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    
    df['num_symbols'] = df['comment_text']\
        .apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    
    df['num_words'] = df['comment_text']\
        .apply(lambda comment: len(comment.split()))
    
    df['num_unique_words'] = df['comment_text']\
        .apply(lambda comment: len(set(w for w in comment.split())))
    
    df['words_vs_unique'] = np.round(df['num_unique_words'] / df['num_words'],4)
    
    df['num_smilies'] = df['comment_text']\
        .apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    
    return df