# tfidf_search
This is the first project assignment of the class <strong>Web Search and Mining</strong> in 2019 spring semester in National Chengchi University(NCCU). I modified some source code (VectorSpace.py, tfidf.py, Parser.py, PorterStemmer.py, amdd util.py) and compose the main program(main.py) to implement the TF-IDF calculation and find the top 10 related documents in these 2048 documents.

## Description
First, user can enter the query. Then, the program will load the whole documents and calculate the term frequency(TF) and TF-IDF. So, we can use the value of TF and TF-IDF to search the most relative document and represent on the screen.
<p>
However, because the program will load the whole documents after enter the query, the process of loading costs a lot of time.
<p>
Note: This main.py cannot run well, so you can delete line 41 in VectorSpace.py to see the top 2 answers(tf+cosine, tf+Euclidean).

## Python Version
Python 3.7.1

## Student
<strong>CHANG Chi-Hung</strong><p>
junior<br>
Dept. in Computer Science<br>
National Chengchi University

## Professor
<strong>TSAI Ming-Feng</strong>

