from whooshIndex import getIndex, fieldList
from whoosh.qparser import MultifieldParser #enable multiple field search
from whoosh import scoring #set up the scoring system
import shutil #for printing purposes
from sys import exit

mode = 0 #search engine mode
mode_min = 1
mode_max = 3

menu = """
--------------------------
0) Exit
1) Normal mode
2) Sentiment Analysis mode
3) Word2Vec mode
--------------------------
"""


#print results of each document found
def print_hit_result(hit):
    terminal_width = shutil.get_terminal_size().columns
    print(f"Review ID: {hit.get('reviewID')}")
    print(f"Review Date: {hit.get('reviewData')}")
    print(f"Author Name: {hit.get('authorName')}")
    print(f"Vehicle Name: {hit.get('vehicleName')}")
    print(f"Review Title: {hit.get('reviewTitle')}")
    print(f"Review Text: {hit.get('reviewText')}")
    print(f"Review Rating: {hit.get('reviewRating')}")
    print("Score:", hit.score)
    print("-" * terminal_width)
    print("\n")


def get_mode():
    while True:
        print(menu)
        mode_choice = input("Choose a mode: ")
        try:
            mode_choice = int(mode_choice)
        except ValueError:
            print(f"Invalid choice: {mode_choice}")
        else:
            if mode_choice == 0:
                exit()
            elif mode_choice < mode_min or mode_choice > mode_max:
                print(f"Invalid choice: {mode_choice}")
            else:
                break
    return mode_choice

            




#main
def main():
    
    index = getIndex("index") #retrieve the index
    searcher = index.searcher(weighting=scoring.TF_IDF()) #set the scoring system
    global mode
    mode = get_mode()

    while True:
        query_str = input("Insert query, press enter to stop: ")
        if not query_str.strip():
            break
        parser = MultifieldParser(fieldList, schema=index.schema) #search throughout all of the fields of the schema
        query = parser.parse(query_str) #parse the query
        results = searcher.search(query, limit=5, scored=True) #set the document limit to 5 and enable the scoring
        for hit in results: #print the documents matching the query
            print_hit_result(hit)

if __name__ == "__main__":
    main()
