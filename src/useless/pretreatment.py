import nltk
from tokenizer import tokenizer

def tweet_tokenizer(tweet_sentence):
    """Function that return a list of token adapted for tweet giving a tweet string"""

    T = tokenizer.TweetTokenizer()
    tokens = T.tokenize(tweet_sentence)
    return tokens

def tokens_stemming(tokens):
    """Function that return a list of stemmed token giving a list of token"""
    
    sno = nltk.stem.SnowballStemmer('english')
    tokens_stemming = [sno.stem(token) for token in tokens]
    return tokens_stemming


def main():
    tweet = "sad:( I'm we've I'll n't So @Ryanair site crashes everytime I try to book - how do they help? Tell me there's nothing wrong &amp; hang up #furious #helpless @SimonCalder"
    tokens = tweet_tokenizer(tweet)
    tokens = tokens_stemming(tokens)
    print(tokens)


if __name__ == "__main__":
    main()
