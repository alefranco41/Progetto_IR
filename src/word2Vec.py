import numpy as np
import pandas as pd
import os
import re
import time
from gensim.models import Word2Vec
from tqdm import tqdm
from globalVariables import CSVdataPath, word2VecModel, word2VecModelPath
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import KeyedVectors

tqdm.pandas()


def preprocessing(reviews):
    processed_array = []
    for review in tqdm(reviews):
        processed = re.sub('[^a-zA-Z0-9 ]', '', review)
        words = processed.split()
        processed_array.append(' '.join([word for word in words if len(word) > 1]))
    return processed_array

def make_model():
    file_paths = [os.path.join(CSVdataPath, file) for file in os.listdir(CSVdataPath) if file.endswith(".csv")]
    train_files, test_files = train_test_split(file_paths, test_size=0.3, random_state=42)
    
    df_train = pd.DataFrame(columns=['Review'])
    df_test = pd.DataFrame(columns=['Review'])


    for file_path in train_files:
        file_path = file_path.lower()
        df = pd.read_csv(file_path, lineterminator='\n')
        df['processed'] = pd.Series(preprocessing(df['Review']))
        df_train = pd.concat([df_train, df[['processed']]], ignore_index=True)


    for file_path in test_files:
        file_path = file_path.lower()
        df = pd.read_csv(file_path, lineterminator='\n')
        df['processed'] = pd.Series(preprocessing(df['Review']))
        df_test = pd.concat([df_test, df[['processed']]], ignore_index=True)
 


    sentences = pd.concat([df_train['processed'], df_test['processed']], axis=0)
    train_sentences = list(sentences.progress_apply(str.split).values)
    model = Word2Vec(sentences=train_sentences, sg=1, vector_size=100, workers=4)
    model.wv.save_word2vec_format(word2VecModelPath)
    
    word2VecModel = KeyedVectors.load_word2vec_format(word2VecModelPath)
    documents_mapping = {}
    for i,sentence in enumerate(sentences):
        words = sentence.split()
        word_vectors = []
        for word in words:
            if word in word2VecModel:
                word_vectors.append(word2VecModel[word])
        if word_vectors:
            mean_vector = np.mean(word_vectors, axis=0)

        documents_mapping[i] = (sentence, mean_vector)

    with open("Word2VecDocumentMapping.bin", "wb") as file:
        pickle.dump(documents_mapping, file)

    return model


if __name__ == "__main__":
    if not word2VecModel:
        model = make_model()
    else:
        model = word2VecModel
