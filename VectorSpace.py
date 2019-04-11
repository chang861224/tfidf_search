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
    idfVector = []

    # Collection of document feedback vectors
    feedbackVectors = []

    # Mapping of vector index to keyword
    vectorKeywordIndex = []
        
    uniqueVocabularyList = {}

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
        self.idfVector = self.makeIdfVector()
        self.documentVectors = [self.makeVector(str(document)) for document in documents]
        self.tfidfVectors = [self.makeTfidfVector(documentVector) for documentVector in self.documentVectors]


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        self.uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in self.uniqueVocabularyList:
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


    def makeIdfVector(self):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        for word in self.uniqueVocabularyList:
            vector[self.vectorKeywordIndex[word]] = tfidf.idf(word, self.bloblist) 
        return vector


    def makeTfidfVector(self, documentVector):
        tfVector = [float(i) for i in documentVector]
        return [a*b for a, b in zip(tfVector, self.idfVector)]

    
    def makeFeedback(self, documentVector, queryVector):
        vector = np.array(queryVector) + 0.5 * np.array(documentVector)
        return vector
    
    
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def buildQueryTfidf(self, termList):
        query = self.makeTfidfVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        return ratings


    def searchTfByCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings
    
    
    def searchTfByEuclidean(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings
    
    
    def searchTfidfByCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryTfidf = self.makeTfidfVector(queryVector)

        ratings = [util.cosine(queryTfidf, tfidfVector) for tfidfVector in self.tfidfVectors]
        return ratings
    
    
    def searchTfidfByEuclidean(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        queryTfidf = self.makeTfidfVector(queryVector)

        ratings = [util.euclidean(queryTfidf, tfidfVector) for tfidfVector in self.tfidfVectors]
        return ratings


    def searchFeedback(self, searchList):
        queryVector = self.buildQueryVector(searchList)
        queryTfidf = self.makeTfidfVector(queryVector)

        feedbackVectors = [self.makeFeedback(documentVector, queryVector) for documentVector in self.documentVectors]
        feedbackTfidf = [self.makeTfidfVector(feedbackVector) for feedbackVector in feedbackVectors]

        ratings = [util.cosine(queryTfidf, feedbackTfidfVector) for feedbackTfidfVector in feedbackTfidf]
        return ratings

