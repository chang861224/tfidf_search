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

    #bloblist = []

    # Collection of document feedback vectors
    #feedbackVectors = []

    # Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents = []):
    #def __init__(self, documents = [], bloblist = []):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)
            #self.build(documents, bloblist)

    def build(self,documents):
    #def build(self,documents, bloblist):
        """ Create the vector space for the passed document strings """
        #self.bloblist = [tb(document) for document in documents]
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]
        #self.tfidfVectors = [self.makeTfidf(document, blob, bloblist) for document, blob in zip(documents, bloblist)]

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

    
    # def makeTfidf(self, wordString, bloblist):
    #def makeTfidf(self, wordString, blob, bloblist):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        #vector = [0] * len(self.vectorKeywordIndex)
        #wordList = self.parser.tokenise(wordString)
        #wordList = self.parser.removeStopWords(wordList)
        #for word in wordList:
        #    vector[self.vectorKeywordIndex[word]] = tfidf.tfidf(word, blob, bloblist)
            #vector[self.vectorKeywordIndex[word]] = {tfidf.tfidf(word, blob, bloblist) for blob, word in zip(bloblist, blob.words)}
        #return vector

    
    #def makeFeedback(self, count, queryVector):
        #vector = np.array(queryVector) + 0.5 * np.array(documentVectors[count])
        #return vector
    
    
    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    #def buildQueryTfidf(self, termList, bloblist):
        #query = self.makeTfidf((" ".join(termList), blob, bloblist) for blob in bloblist)
        #return query


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
    
    
    #def searchTfidfByCosine(self, searchList, bloblist):
        #""" search for documents that match based on a list of terms """
        #queryVector = self.buildQueryTfidf(searchList, bloblist)

        #ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        #return ratings
    
    
    #def searchTfidfByEuclidean(self, searchList, bloblist):
        #""" search for documents that match based on a list of terms """
        #queryVector = self.buildTfidf(searchList, bloblist)

        #ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        #return ratings


    #def searchFeedback(self, searchList, bloblist):
        #queryVector = self.buildQueryVector(searchList)

        #feedbackVectors = [self.makeFeedback(count, queryVector) for count in range(len(documentVectors))]

        #feedbackTfidfVectors = self.buildQueryTfidf(searchList, bloblist)

        #ratings = [util.cosine(feedbackTfidfVectors, documentVector) for documentVector in self.documentVectors]
        #return ratings

