from whoosh.qparser import SimpleParser, MultifieldParser #enable multiple field search
from whoosh.analysis import Tokenizer
from sys import exit
from os import system
import globalVariables
from word2Vec import preprocessing #preprocess the query for word2vec mode
import pandas as pd
import numpy as np

menu = """
0) Exit
1) Full-text search mode
2) Sentiment Analysis mode
3) Word2Vec mode
"""


#print results of each document found
def print_hit_result(hit, reviews, sentiment):
    if reviews:
        print(f"Review ID: {hit.get('reviewID')}")
        print(f"Vehicle Name: {hit.get('rawVehicleName')}")
        print(f"Review Date: {hit.get('reviewData')}")
        print(f"Author Name: {hit.get('authorName')}")
        print(f"Review Title: {hit.get('reviewTitle')}")
        print(f"Review Text: {hit.get('reviewText')}")
        print(f"Review Rating: {hit.get('reviewRating')}")
        print(globalVariables.str_separator)
    else:
        print(f"Vehicle Name: {hit.get('vehicleName')}")
    
    if sentiment:
        print(f"Sentiment Score: {hit.get('sentimentScore')}")



def get_mode():
    while True:
        print(globalVariables.str_separator + menu + globalVariables.str_separator)
        mode_choice = input("Choose a mode: ")
        try:
            mode_choice = int(mode_choice)
        except ValueError:
            system('clear')
            print(f"Invalid choice: {mode_choice}")
        else:
            if mode_choice == 0:
                exit()
            elif mode_choice < globalVariables.mode_min or mode_choice > globalVariables.mode_max:
                system('clear')
                print(f"Invalid choice: {mode_choice}")  
            else:
                break
    system('clear')
    return mode_choice

            


def parse_query(query_str):
    if len(query_str.split()) == 1:
        for word in globalVariables.review_words:
            if word in query_str:
                return query_str, True    
        return query_str, False
    
    reviews = False
    query = globalVariables.custom_tokenizer.tokenize(query_str)
    filtered_tokens = [word for word in query if word.lower() not in globalVariables.stop_words]
    for word in globalVariables.review_words:
        if word in filtered_tokens:
            filtered_tokens.remove(word)
            reviews = True

    if not reviews:
        for word in globalVariables.review_words:
            if word in query_str:
                return query_str, True
            
    if reviews:
        return " ".join(filtered_tokens), reviews
    return query_str, reviews
    

def full_text_mode(sentiment):
    if not sentiment:
        index = globalVariables.index
        fieldList = globalVariables.fieldList
    else:
        index = globalVariables.sentimentIndex
        fieldList = globalVariables.sentimentFieldList
    
    while True: 
        query_str = input("Insert query, press enter to stop: ")
        system('clear')
        print(f"Mode: {globalVariables.modes[mode]}")
        if not query_str.strip():
            break
        query_str, reviews = parse_query(query_str)
        if not sentiment:
            reviews = True

        if reviews:
            parser = MultifieldParser(fieldList, schema=index.schema) #search throughout all of the fields of the schema
        else:
            parser = SimpleParser("vehicleName", schema=index.schema) #search throughout all of the fields of the schema
        
        
        query = parser.parse(query_str) #parse the query
        searcher = index.searcher() #set the scoring system
        results = searcher.search(query, limit=globalVariables.limit, scored=True) #set the document limit to 5 and enable the scoring
        if results: #if some results are retrieved
            if not reviews: #if we are searching about text items (vehicle names and not reviews)
                extended_results = []
                #make sure that we don't get any duplicate
                i = 2
                while True:
                    searcher = index.searcher() #set the scoring system
                    new_results = searcher.search(query, limit=globalVariables.limit*i, scored=True)
                    extended_results.extend(new_results)
                    filtered_results = set([" ".join(hit.get('vehicleName').split()[0:2]) for hit in extended_results])
                    if len(filtered_results) >= min(globalVariables.limit, len(results)) or len(extended_results) >= 50000: #to avoid an infinite loop we set a limit to "extended_results"
                        break
                    i+=1

                unique_results = {}
                for hit in extended_results:
                    vehicle_name_prefix = " ".join(hit.get('vehicleName').split()[0:2])
                    if vehicle_name_prefix not in unique_results:
                        unique_results[vehicle_name_prefix] = hit

                #make sure that the length of the retrieved vehicle list matches the limit           
                min_len = min(globalVariables.limit, len(results))
                results = list(unique_results.values())[:min_len] 

            #if the user is searching through reviews (and not vehicle names) we leave "results" as it is
            n = 1
            print(f"Results for '{query_str}': " + "\n")
            for hit in results: #print the documents matching the query
                print(f"Hit result #{n}")
                print_hit_result(hit, reviews, sentiment)
                n+=1
        else:
            print(f"No results found for '{query_str}' " + "\n")



def word2Vec_mode():
    while True:
        search_results = []
        query_str = input("Insert query, press enter to stop: ")
        system('clear')
        print(f"Mode: {globalVariables.modes[mode]}")
        if not query_str.strip():
            break
        query_tokens = preprocessing(pd.Series(query_str))[0].split()
        query_embeddings = [globalVariables.word2VecModel[token] for token in query_tokens if token in globalVariables.word2VecModel]
        if not query_embeddings:
            print("No valid tokens found in the query.")
            continue
        query_vector = np.mean([embedding for embedding in query_embeddings], axis=0)
        for doc, doc_vector in globalVariables.word2vec_vectors.items():
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            if isinstance(similarity, np.ndarray):
                continue
            search_results.append((doc, similarity))

        search_results.sort(key=lambda x: x[1], reverse=True)
        parser = SimpleParser("reviewID", schema=globalVariables.index.schema) #search throughout all of the fields of the schema
        print(f"Results for '{query_str}':\n")
        for i, (doc, similarity) in enumerate(search_results[:globalVariables.limit], 1):
            print(f"Hit result #{i}")
            print(f"Similarity: {similarity}\n")
            query = parser.parse(str(doc)) #parse the query
            searcher = globalVariables.index.searcher() #set the scoring system
            result = searcher.search(query, limit=1, scored=True) #set the document limit to 5 and enable the scoring
            print_hit_result(result[0], True, False)
#main
def main():
    global mode
    mode = get_mode()
    print(f"Mode: {globalVariables.modes[mode]}")
    if mode == 1:
        full_text_mode(False)
    elif mode == 2:
        full_text_mode(True)
    else:
        word2Vec_mode()
        


if __name__ == "__main__":
    main()
