import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')

class TikTokAnalyzer():
    def __init__(self):
        self.data = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
    
    def clean(self, text):
        stop_words = set(stopwords.words('english'))
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stop_words]
        text = ' '.join(text)
        return text

    def eda_process(self):
        print(self.data.head())

        # Only keep necessary columns
        self.data = self.data[['content', 'score']]
        print(self.data.head())

        # Checking for null values
        print(self.data.isnull().sum())

        # Drop null values
        self.data = self.data.dropna()

        # Clean the content column
        print('Formating and cleaning Column contet for sentiments analysis')
        self.data['content'] = self.data['content'].apply(self.clean)

        # Analyze the ratings
        ratings = self.data['score'].value_counts()
        numbers = ratings.index
        quantity = ratings.values
        import plotly.express as px
        figure = px.pie(self.data, values=quantity, names=numbers, hole=0.5)
        figure.show()

        # Generate a word cloud
        text = ' '.join(i for i in self.data.content)
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # Sentiment analysis
        sentiments = SentimentIntensityAnalyzer()
        self.data['Positive'] = [sentiments.polarity_scores(i)['pos'] for i in self.data['content']]
        self.data['Negative'] = [sentiments.polarity_scores(i)['neg'] for i in self.data['content']]
        self.data['Neutral']  = [sentiments.polarity_scores(i)['neu'] for i in self.data['content']]
        self.data = self.data[['content', 'Positive', 'Negative', 'Neutral']]
        print(self.data.head())

        # Positive word cloud
        positive = ' '.join(self.data.loc[self.data['Positive'] > self.data['Negative'], 'content'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(positive)
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # Negative word cloud
        negative = ' '.join(self.data.loc[self.data['Negative'] > self.data['Positive'], 'content'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(negative)
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        
        print('EDA process finished')

# Usage example
if __name__ == '__main__':
    analyzer = TikTokAnalyzer()
    analyzer.load_data('D:\\Datascienceprojects\\Data_Analytics\\TikTok_Reviews_Sentiment_Analysis\\tiktok_google_play_reviews.csv')
    analyzer.eda_process()
