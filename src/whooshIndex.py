from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, NUMERIC
from whoosh.index import create_in, open_dir
from whoosh.writing import BufferedWriter
from os import path, mkdir, listdir, remove
from csv import reader #needed to opens csv files
from sys import exit 
from datetime import datetime #needed to convert text into dates
import globalVariables


#creating Woosh scheme for indexing
schema = Schema(
    reviewID = NUMERIC(stored=True),
    reviewDate = DATETIME(stored=True),
    authorName = TEXT(stored=True),
    rawVehicleName = TEXT(stored=True),
    reviewTitle = TEXT(stored=True),
    reviewText = TEXT(stored=True),
    reviewRating = NUMERIC(stored=True),
    vehicleName  = TEXT(stored=True)
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


currentID = 0

#store each row of each csv file to the index as a document
def addToIndex(index, csvFile, filename):
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


#for every csv file, add its rows to the index
def populateIndex(index, dataDirectory, limit):
    try:
        dir = listdir(dataDirectory)
    except FileNotFoundError:
        print(f"Error, directory '{dataDirectory}' not found")
        exit()
    else:
        i=0
        for filename in dir:
            file_path = path.join(dataDirectory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                csvFile = reader(file)
                next(csvFile) #skip the first line of the csv file since it doesn't contain data
                addToIndex(index, csvFile, filename) #function called for every csv file in the specified directory
                i+=1
            if limit != 0:
                if i == limit:
                    break


#if the module 'whooshindex' is ran by itself
#create a new index containing the csv files in the 'CSVdata' directory 
if __name__ == "__main__":
    limit = 2 #set this to 0 if you want to index all the files
    dataDirectory = globalVariables.CSVdataPath
    indexDirectory = globalVariables.indexPath
    index = getIndex(indexDirectory)
    populateIndex(index, dataDirectory, limit)