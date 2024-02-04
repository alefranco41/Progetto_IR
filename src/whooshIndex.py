from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, NUMERIC
from whoosh.index import create_in, open_dir
from whoosh.writing import BufferedWriter
from os import path, mkdir, listdir
from csv import reader #needed to opens csv files
from sys import exit 
from datetime import datetime #needed to convert text into dates

fieldList = ["reviewData", "authorName", "vehicleName", "reviewTitle", "reviewText", "reviewRating"]


#creating Woosh scheme for indexing
schema = Schema(
    reviewID = NUMERIC(stored=True),
    reviewData = DATETIME(stored=True),
    authorName = TEXT(stored=True),
    vehicleName = TEXT(stored=True),
    reviewTitle = TEXT(stored=True),
    reviewText = TEXT(stored=True),
    reviewRating = NUMERIC(stored=True)
)

def getIndex(dir):
    #create the index with the Woosh scheme if it doesn't already exist
    if not path.exists(dir):
        mkdir(dir)
        ix = create_in(dir, schema)
        print("Index created")
    else:
        #if the index already exists, open it.
        ix = open_dir(dir)
        print("Index retrieved")
    return ix



#store each row of each csv file to the index as a document
def addToIndex(index, csvFile, filename):
    writer = index.writer()
    for row in csvFile:
        try:
            if len(row) == 7: #skip the rows not matching the scheme
                #convert the first column to an integer
                reviewID = int(row[0])
                
                #convert the second column as a valid date
                reviewData_str = row[1].replace(" on ", "")
                reviewData_str = reviewData_str.replace(" (PDT)", "").replace(" (PST)", "")
                reviewData = datetime.strptime(reviewData_str, "%m/%d/%y %H:%M %p")
                
                authorName = row[2]
                vehicleName = row[3]
                reviewTitle = row[4]
                reviewText = row[5]
                reviewRating = float(row[6])
                
                #add the row to the index as a document
                writer.add_document(reviewID=reviewID, reviewData=reviewData, authorName=authorName, vehicleName=vehicleName, reviewTitle=reviewTitle, reviewText=reviewText, reviewRating=reviewRating)
                print(f"The document {filename}:{reviewID} has been added to the index")
        except Exception as e:
            #if an error occurs, skip to the next row
            print(f"Error adding document: {e}")
            continue
    #save the changes to the index
    writer.commit()


#for every csv file, add its rows to the index
def populateIndex(index, dataDirectory):
    try:
        dir = listdir(dataDirectory)
    except FileNotFoundError:
        print(f"Error, directory '{dataDirectory}' not found")
        exit()
    else:
        for filename in dir:
            file_path = path.join(dataDirectory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                csvFile = reader(file)
                next(csvFile) #skip the first line of the csv file since it doesn't contain data
                addToIndex(index, csvFile, filename) #function called for every csv file in the specified directory


#if the module 'whooshindex' is ran by itself
#create a new index containing the csv files in the 'CSVdata' directory 
if __name__ == "__main__":
    dataDirectory = "CSVdata"
    indexDirectory = "index"

    index = getIndex(indexDirectory)
    populateIndex(index, dataDirectory)