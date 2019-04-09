from __future__ import division, unicode_literals
from Parser import Parser
import util
import tfidf
import numpy as np
from textblob import TextBlob as tb

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    # Collection of document term vectors
    documentVectors = []

    tfidfVectors = []
    bloblist = []

    # Collection of document feedback vectors
    feedbackVectors = []

    # Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents = []):
        self.documentVectors=[]
        self.bloblist = documents
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(str(document) for document in documents)
        self.documentVectors = [self.makeVector(str(document)) for document in documents]
        #self.tfidfVectors = [self.makeTfidf(str(document)) for document in documents]

        #print self.vectorKeywordIndex
        #print self.documentVectors


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector


    def makeTfidf(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] = self.calTfidf(word, self.bloblist)
        return vector


    def calTfidf(self, word, bloblist):
        for blob in enumerate(bloblist):
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        return scores[word]

    
    def makeFeedback(self, count, queryVector):
        vector = np.array(queryVector) + 0.5 * np.array(documentVectors[count])
        return vector
    
    
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def buildQueryTfidf(self, termList):
        query = self.makeTfidf(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def searchTfByCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    
    def searchTfByEuclidean(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    
    def searchTfidfByCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryTfidf(searchList)

        print(queryVector)
        print(self.tfidfVectors)

        ratings = [util.cosine(queryVector, tfidfVector) for tfidfVector in self.tfidfVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    
    def searchTfidfByEuclidean(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryTfidf(searchList)

        ratings = [util.euclidean(queryVector, tfidfVector) for tfidfVector in self.tfidfVectors]
        #ratings.sort(reverse=True)
        return ratings


    def searchFeedback(self, searchList):
        queryVector = self.buildQueryVector(searchList)

        feedbackVectors = [self.makeFeedback(documentVector, queryVector) for documentVector in range(len(documentVectors))]

        feedbackTfidfVectors = self.buildQueryTfidf(searchList)

        ratings = [util.cosine(feedbackTfidfVectors, tfidfVector) for tfidfVector in self.tfidfVectors]
        return ratings

