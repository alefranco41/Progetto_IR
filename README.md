# Progetto_IR

Usage:
1) Install the dependencies listed in the "requirements.txt" file
2) Unpack CSVdata.rar inside /src directory
3) python3 whooshIndex.py (inside src directory) -> stores the Whoosh index in /index
4) python3 word2Vec.py (inside src directory) -> stores the trained word2vec model in /word2Vec
5) python3 sentimentAnalysis.py (inside src directory) -> stores the trained sentiment model in /sentiment
5) wait (you can set the variable "limit" to a low value if you don't want to index all the files)
6) python3 main.py (inside src directory)
