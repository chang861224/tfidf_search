import os
import tfidf
import math
import numpy as np
from VectorSpace import VectorSpace
from textblob import TextBlob as tb
from pprint import pprint



print("Please enter the query: ")
x = input()



path = "documents/"

ID = []
documents = []
#bloblist = []

for r, d, f in os.walk(path):
    for file in f:
        name = os.path.basename(file)
        i = open(path + name)
        ID.append(os.path.splitext(name)[0])
        documents.append(tb(i.read()))

#for blob in range(len(documents)):
#    bloblist.append(tb(documents[blob]))

VectorSpace = VectorSpace(documents)
#VectorSpace = VectorSpace(documents, bloblist)




#####################################
##                                 ##
##  TF Weight + Cosine Similarity  ##
##                                 ##
#####################################
scores = (ID, VectorSpace.searchTfByCosine([x]))
scores = np.transpose(scores)
scores = sorted(scores, key = lambda x: x[1], reverse = True)

print("----------------")
print("Term Frequency Weight + Cosine Similarity:")
print("")
print("DocID\tScore")
print("------- -------")

for DocID, score in scores[:10]:
    print("{}\t{:.6f}".format(DocID, round(float(score), 6)))



print("")



######################################
##                                  ##
##  TF Weight + Euclidean Distance  ##
##                                  ##
######################################
scores = (ID, VectorSpace.searchTfByEuclidean([x]))
scores = np.transpose(scores)
scores = sorted(scores, key = lambda x: float(x[1]), reverse = False)

print("----------------")
print("Term Frequency Weight + Euclidean Distance:")
print("")
print("DocID\tScore")
print("------- -------")

for DocID, score in scores[:10]:
    print("{}\t{:.6f}".format(DocID, round(float(score), 6)))


'''
print("")



#########################################
##                                     ##
##  TF-IDF Weight + Cosine Similarity  ##
##                                     ##
#########################################
scores = (ID, VectorSpace.searchTfidfByCosine([x], bloblist))
scores = np.transpose(scores)
scores = sorted(scores, key = lambda x: x[1], reverse = True)

print("----------------")
print("TF-IDF Weight + Cosine Similarity:")
print("")
print("DocID\tScore")
print("------- -------")

for DocID, score in scores[:10]:
    print("{}\t{:.6f}".format(DocID, round(float(score), 6)))



print("")



##########################################
##                                      ##
##  TF-IDF Weight + Euclidean Distance  ##
##                                      ##
##########################################
scores = (ID, VectorSpace.searchTfidfByEuclidean([x], bloblist))
scores = np.transpose(scores)
scores = sorted(scores, key = lambda x: float(x[1]), reverse = False)

print("----------------")
print("TF-IDF Weight + Euclidean Distance:")
print("")
print("DocID\tScore")
print("------- -------")

for DocID, score in scores[:10]:
    print("{}\t{:.6f}".format(DocID, round(float(score), 6)))



print("")



##########################################################
##                                                      ##
##  Feedback Query + TF-IDF Weight + Cosine Similarity  ##
##                                                      ##
##########################################################
scores = (ID, VectorSpace.searchFeedback([x], bloblist))
scores = np.transpose(scores)
scores = sorted(scores, key = lambda x: x[1], reverse = True)

print("----------------")
print("Feedback Query + TF-IDF Weight + Cosine Similarity:")
print("")
print("DocID\tScore")
print("------- -------")

for DocID, score in scores[:10]:
    print("{}\t{:.6f}".format(DocID, round(float(score), 6)))
'''
