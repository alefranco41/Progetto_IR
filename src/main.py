from whooshIndex import getIndex, fieldList
from whoosh.qparser import SimpleParser, MultifieldParser #enable multiple field search
from whoosh import scoring #set up the scoring system
import shutil #for printing purposes
from sys import exit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from os import system

terminal_width = shutil.get_terminal_size().columns
str_separator = "-" * terminal_width

modes = {1: 'Full text search', 2:'Sentiment analysis', 3:'Word2Vec'}
mode = 0  #chosen search engine mode
mode_min = 1
mode_max = 3
limit = 5 #max amount of documents to show

words_to_remove = set(["car", "cars", "vehicle", "vehicles", "search", "find", "named"])
stop_words = set(stopwords.words('english')).union(words_to_remove)
review_words = ["review", "reviews", "opinion", "opinions", "reviewed", "rating"]


custom_token_pattern = r'\b\w+(?::\w+)?\b'
custom_tokenizer = RegexpTokenizer(custom_token_pattern)



menu = """
0) Exit
1) Normal mode
2) Sentiment Analysis mode
3) Word2Vec mode
"""


#print results of each document found
def print_hit_result(hit, reviews):
    if reviews:
        print(f"Vehicle Name: {hit.get('rawVehicleName')}")
        print(f"Review Date: {hit.get('reviewData')}")
        print(f"Author Name: {hit.get('authorName')}")
        print(f"Review Title: {hit.get('reviewTitle')}")
        print(f"Review Text: {hit.get('reviewText')}")
        print(f"Review Rating: {hit.get('reviewRating')}")
        print(str_separator)
    else:
        print(f"Vehicle Name: {hit.get('vehicleName')}")



def get_mode():
    while True:
        print(str_separator + menu + str_separator)
        mode_choice = input("Choose a mode: ")
        try:
            mode_choice = int(mode_choice)
        except ValueError:
            system('clear')
            print(f"Invalid choice: {mode_choice}")
        else:
            if mode_choice == 0:
                exit()
            elif mode_choice < mode_min or mode_choice > mode_max:
                system('clear')
                print(f"Invalid choice: {mode_choice}")  
            else:
                break
    system('clear')
    return mode_choice

            


def parse_query(query_str):
    if len(query_str.split()) == 1:
        for word in review_words:
            if word in query_str:
                return query_str, True    
        return query_str, False
    
    reviews = False
    query = custom_tokenizer.tokenize(query_str)
    filtered_tokens = [word for word in query if word.lower() not in stop_words]
    for word in review_words:
        if word in filtered_tokens:
            filtered_tokens.remove(word)
            reviews = True

    if not reviews:
        for word in review_words:
            if word in query_str:
                return query_str, True
            
    if reviews:
        return " ".join(filtered_tokens), reviews
    return query_str, reviews
    


#main
def main():
    index = getIndex("index") #retrieve the index
    searcher = index.searcher(weighting=scoring.TF_IDF()) #set the scoring system
    global mode
    mode = get_mode()
    print(f"Mode: {modes[mode]}")
    while True:
        query_str = input("Insert query, press enter to stop: ")
        system('clear')
        print(f"Mode: {modes[mode]}")
        if not query_str.strip():
            break
        query_str, reviews = parse_query(query_str)
        if reviews:
            parser = MultifieldParser(fieldList, schema=index.schema) #search throughout all of the fields of the schema
        else:
            parser = SimpleParser("vehicleName", schema=index.schema) #search throughout all of the fields of the schema
        query = parser.parse(query_str) #parse the query
        results = searcher.search(query, limit=limit, scored=True) #set the document limit to 5 and enable the scoring
        if results: #if some results are retrieved
            if not reviews: #if we are searching about text items (vehicle names and not reviews)
                extended_results = []
                #make sure that we don't get any duplicate
                i = 2
                while True:
                    new_results = searcher.search(query, limit=limit*i, scored=True)
                    extended_results.extend(new_results)
                    filtered_results = set([" ".join(hit.get('vehicleName').split()[0:2]) for hit in extended_results])
                    if len(filtered_results) >= min(limit, len(results)) or len(extended_results) >= 1000: #to avoid an infinite loop we set a limit to "extended_results"
                        break
                    i+=1

                unique_results = {}
                for hit in extended_results:
                    vehicle_name_prefix = " ".join(hit.get('vehicleName').split()[0:2])
                    if vehicle_name_prefix not in unique_results:
                        unique_results[vehicle_name_prefix] = hit

                #make sure that the length of the retrieved vehicle list matches the limit           
                min_len = min(limit, len(results))
                results = list(unique_results.values())[:min_len] 

            #if the user is searching through reviews (and not vehicle names) we leave "results" as it is
            n = 1
            print(f"Results for '{query_str}': " + "\n")
            for hit in results: #print the documents matching the query
                print(f"Hit result #{n}")
                print_hit_result(hit, reviews)
                n+=1
        else:
            print(f"No results found for '{query_str}' " + "\n")

        


if __name__ == "__main__":
    main()
