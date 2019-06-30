"""
James Getsinger
University of North Carolina Charlotte
Study Report
"""


import pandas as pd
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from textblob import TextBlob
import csv

stop = stopwords.words('english')

def getSentiment():
    
    with open('yelp.csv','r') as csvinput:
        with open('output.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            rows = csv.reader(csvinput)
            all = []
            for row in rows:
                sentence = row[7]
                blob = TextBlob(sentence)
                row.append(blob.sentiment.polarity)
                all.append(row)
    
            for row in rows:
                row.append(row[9])
                all.append(row)
    
            writer.writerows(all)
            
def getBigrams(rev):
    
    with open('finaldata.csv','r') as csvinput:    
        df = pd.read_csv(csvinput)
        
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        df['text'] = df['text'].str.replace(r'[^\w\s\d+]',' ').str.lower()
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df['text'] = df['text'].apply(word_tokenize)
    
        df['text'] = [tuple(x) for x in df['text']]
        df['text'] = df.apply(lambda df: BigramCollocationFinder.from_words(df['text']),axis=1)
        df['text'].apply(lambda df: df.apply_freq_filter(1))
        
        #list of bigrams that start with positive word followed by item searching for
        whitelist = ["best", "nice", "good", "wonderful", "great", "amazing"]        
        whitelist1 = [rev]  
        df['text'].apply(lambda df: df.apply_ngram_filter(lambda w1, w2: w1 not in whitelist ))          
        df['text'].apply(lambda df: df.apply_ngram_filter(lambda w1, w2: w2 not in whitelist1 )) 
        #find the likelihood ratio that bigram appears in text of business         
        df['text'] = df['text'].apply(lambda df: df.score_ngrams(bigram_measures.likelihood_ratio))
        df = df.sort_values(by=['text'], ascending=False)
        #provide recommendations of businesses with at least .285(average sentiment
        #across all businesses) minimum sentiment out of the top 5 returned results
        for i in range(5):
            if ((df['sentiment'].iloc[i]) > .285):
                print("\n For good " + rev + " in Phoenix, AZ, we recommend: " + df['business_name'].iloc[i])

    
if __name__ == '__main__':
    #get sentiment is used to first get scores across category.  Then,
    #sentiment is averaged across businesses in excel to get overall sentiment
    #score of business
    #getSentiment()
    print("Recommending business in same cateogry")
    #This would be used for customer leaving bad review about business in category.
    #They will be recommended businesses that excel at item they did not like at
    #location
    rev = input("What item did you not like about the location? Enter single word response (i.e. tea) \n")
    getBigrams(rev)

    