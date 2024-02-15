import numpy as np
import pandas as pd
import os
import re
from gensim.models import Word2Vec
from tqdm import tqdm
from globalVariables import CSVdataPath, word2VecModel, word2VecModelPath, word2VecIndexPath
from sklearn.model_selection import train_test_split
from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, NUMERIC
from whoosh.index import create_in, open_dir
from csv import reader #needed to opens csv files
from sys import exit 
from datetime import datetime #needed to convert text into dates
import pickle

#creating Woosh scheme for indexing
schema = Schema(
    reviewID = NUMERIC(stored=True),
    reviewDate = DATETIME(stored=True),
    authorName = TEXT(stored=True),
    rawVehicleName = TEXT(stored=True),
    reviewTitle = TEXT(stored=True),
    reviewText = TEXT(stored=True),
    reviewRating = NUMERIC(stored=True),
    vehicleName  = TEXT(stored=True),
)

def getIndex(dir):
    #create the index with the Woosh scheme if it doesn't already exist
    if not os.path.exists(dir):
        os.mkdir(dir)
        ix = create_in(dir, schema)
        print("Index created")
    else:
        #if the index already exists, open it.
        ix = open_dir(dir)
        print("Index retrieved")
    return ix


currentID = 0

#store each row of each csv file to the index as a document
def addToIndex(index, csvFile, filename, model):
    vectors = {}
    global currentID
    writer = index.writer()
    for row in csvFile:
        try:
            if len(row) == 7: #skip the rows not matching the scheme
                #convert the first column to an integer
                reviewID = int(row[0]) + currentID
                
                #convert the second column as a valid date
                reviewDate_str = row[1].replace(" on ", "")
                reviewDate_str = reviewDate_str.replace(" (PDT)", "").replace(" (PST)", "")
                reviewDate = datetime.strptime(reviewDate_str, "%m/%d/%y %H:%M %p")
                
                
                try:
                    year = int(row[3].split()[0])
                except Exception:
                    vehicleName = row[3]
                else:
                    vehicleName = row[3].removeprefix(str(year) + ' ')
                

                for pattern in ["1dr", "2dr", "3dr", "4dr", "5dr"]:
                    vehicleName = vehicleName.replace(pattern, "")
                
                splittedVehicleName = vehicleName.split()
                
                if splittedVehicleName[-1][-1] == ")":
                    for index, subStr in enumerate(splittedVehicleName):
                        if subStr[0] == "(":
                            start_index = index
                            splittedVehicleName = splittedVehicleName[:start_index]
                            vehicleName = " ".join(splittedVehicleName)
                            break
                
                            
                tmp_set = set()
                tmp_list = list()
                for name in splittedVehicleName:
                    if name not in tmp_set:
                        tmp_list.append(name)
                        tmp_set.add(name)
                
                vehicleName = " ".join(tmp_list)

                authorName = row[2]
                reviewTitle = row[4]
                reviewText = row[5]
                reviewRating = float(row[6])
                
                review_tokens = preprocessing(pd.Series(reviewText))[0].split()
                review_embeddings = [model[token] for token in review_tokens if token in model]
                review_vector = np.mean([embedding for embedding in review_embeddings], axis=0)
                vectors[reviewID] = review_vector

                #add the row to the index as a document
                writer.add_document(reviewID=reviewID, reviewDate=reviewDate, authorName=authorName, rawVehicleName=row[3], reviewTitle=reviewTitle, reviewText=reviewText, reviewRating=reviewRating, vehicleName=vehicleName)
                print(f"The document {filename}:{reviewID} has been added to the index")
        except Exception as e:
            #if an error occurs, skip to the next row
            print(f"Error adding document: {e}")
            continue
    #save the changes to the index
    writer.commit()
    currentID += reviewID
    return vectors


#for every csv file, add its rows to the index
def populateIndex(index, dataDirectory, limit, model):
    all_vectors = {}
    try:
        dir = os.listdir(dataDirectory)
    except FileNotFoundError:
        print(f"Error, directory '{dataDirectory}' not found")
        exit()
    else:
        i=0
        for filename in dir:
            file_path = os.path.join(dataDirectory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                csvFile = reader(file)
                next(csvFile) #skip the first line of the csv file since it doesn't contain data
                doc_vectors = addToIndex(index, csvFile, filename, model) #function called for every csv file in the specified directory
                all_vectors = {**all_vectors, **doc_vectors}
                i+=1
            if limit != 0:
                if i == limit:
                    break
    return all_vectors


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
    
    


if __name__ == "__main__":
    if not word2VecModel:
        model = make_model()
    else:
        model = word2VecModel
    
    limit = 2 #set this to 0 if you want to index all the files
    dataDirectory = CSVdataPath
    indexDirectory = word2VecIndexPath
    index = getIndex(indexDirectory)
    all_vectors = populateIndex(index, dataDirectory, limit, model)
    with open("word2vec_vectors.txt", "wb") as file:
        pickle.dump(all_vectors, file)