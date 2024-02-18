from whoosh.qparser import SimpleParser, MultifieldParser #enable multiple field search
from whoosh.qparser.dateparse import DateParserPlugin
from whoosh.analysis import Tokenizer
from whoosh.scoring import BM25F
from sys import exit
from os import system
import globalVariables
from word2Vec import preprocessing #preprocess the query for word2vec mode
import pandas as pd
import numpy as np
from collections import OrderedDict


class MyWeighting(BM25F):
    use_final = True
    def final(self, searcher, docnum, score):
        sentiment_score = searcher.stored_fields(docnum).get("sentimentScore", 0.5)
        sentiment_label = searcher.stored_fields(docnum).get("sentimentLabel", "neutral")
        if sentiment_label == "negative":
            sentiment_score = 1 - sentiment_score
        sentiment_factor = sentiment_score * 2 - 1  
        adjusted_score = score * (1 + sentiment_factor)  
        return adjusted_score

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
        print(f"Review Date: {hit.get('reviewDate')}")
        print(f"Author Name: {hit.get('authorName')}")
        print(f"Review Title: {hit.get('reviewTitle')}")
        print(f"Review Text: {hit.get('reviewText')}")
        print(f"Review Rating: {hit.get('reviewRating')}")
    else:
        print(f"Vehicle Name: {hit.get('vehicleName')}")
    
    if sentiment:
        print(f"Sentiment Score: {hit.get('sentimentScore')}")
        print(f"Sentiment Label: {hit.get('sentimentLabel')}")

    print(globalVariables.str_separator)



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
    tokens = query_str.split()
    tokens_to_keep = []

    for token in tokens[:]:  # Iterate over a copy of tokens to avoid modifying it directly
        for field in globalVariables.sentimentFieldList:
            if field in token or token in ["OR", "or", "AND", "and"]:
                tokens_to_keep.append(token)

    # Filter out tokens that need to be removed
    tokens_to_keep = list(OrderedDict.fromkeys(tokens_to_keep))
    tokens = [token for token in tokens if token not in tokens_to_keep]
    tokens_to_keep_str = " ".join(tokens_to_keep) + " "


    tokens = " ".join(tokens).lower().split()
    filtered_tokens = [token for token in tokens if token not in globalVariables.stop_words]
    reviews = False
    for word in globalVariables.review_words:
        if word in filtered_tokens:
            reviews = True
            filtered_tokens.remove(word)   
    if reviews:
        return tokens_to_keep_str + " ".join(filtered_tokens), True
    
    for vehicleWord in globalVariables.uniqueVehicleWords:
        if vehicleWord in filtered_tokens:
            return tokens_to_keep_str + " ".join(filtered_tokens), False

    
    if len(tokens_to_keep) == 1 and "vehicleName" in tokens_to_keep[0] and len(tokens) == 0:
        return tokens_to_keep_str + " ".join(filtered_tokens), False
    
    return tokens_to_keep_str + " ".join(filtered_tokens), True

    

def full_text_mode(sentiment):
    if not sentiment:
        index = globalVariables.sentimentIndex
        fieldList = globalVariables.sentimentFieldList
        searcher = index.searcher() #set the scoring system
    else:
        index = globalVariables.sentimentIndex
        fieldList = globalVariables.sentimentFieldList
        searcher = index.searcher(weighting=MyWeighting)

    
    while True: 
        query_str = input("Insert query, press enter to stop: ")
        system('clear')
        print(f"Mode: {globalVariables.modes[mode]}")
        if not query_str.strip():
            break
        query_str, reviews = parse_query(query_str)
        
        if reviews:
            parser = MultifieldParser(fieldList, schema=index.schema) #search throughout all of the fields of the schema
        else:
            parser = SimpleParser("vehicleName", schema=index.schema) #search throughout all of the fields of the schema
        
        parser.add_plugin(DateParserPlugin())

        query = parser.parse(query_str) #parse the query
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
                    if len(filtered_results) >= min(globalVariables.limit, len(results)) or len(extended_results) >= 1000: #to avoid an infinite loop we set a limit to "extended_results"
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
        

        query_str, reviews = parse_query(query_str)

        tokens = query_str.split()
        tokens_to_keep = []

        for token in tokens[:]:  # Iterate over a copy of tokens to avoid modifying it directly
            for field in globalVariables.sentimentFieldList:
                if field in token or token in ["OR", "or", "AND", "and"]:
                    tokens_to_keep.append(token)

    # Filter out tokens that need to be removed
        tokens_to_keep = list(OrderedDict.fromkeys(tokens_to_keep))
        tokens = [token for token in tokens if token not in tokens_to_keep]
        tokens_to_keep_str = " ".join(tokens_to_keep) + " "


        query_tokens = preprocessing(pd.Series(query_str))[0].split()
        query_embeddings = [globalVariables.word2VecModel[token] for token in query_tokens if token in globalVariables.word2VecModel]

        query_embeddings = []
        print(tokens)
        for token in tokens_to_keep:
            try:
                if token.split(":")[1] in globalVariables.word2VecModel:
                    query_embeddings.append(globalVariables.word2VecModel[token.split(":")[1]])
            except IndexError:
                continue
            
        for token in tokens:
            if token in globalVariables.word2VecModel:
                query_embeddings.append(globalVariables.word2VecModel[token])
        
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
        parser = MultifieldParser(globalVariables.sentimentFieldList, schema=globalVariables.sentimentIndex.schema) #search throughout all of the fields of the schema
        print(f"Results for '{query_str}':\n")
        for i, (doc, similarity) in enumerate(search_results[:globalVariables.limit], 1):
            query_str = f"reviewID:{str(doc)}"
            query = parser.parse(query_str) #parse the query
            searcher = globalVariables.index.searcher() #set the scoring system
            result = searcher.search(query, limit=None, scored=True) #set the document limit to 5 and enable the scoring
            try:
                print_hit_result(result[0], reviews, False)
                print(f"Hit result #{i}")
                print(f"Similarity: {similarity}\n")
            except IndexError:
                continue
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
