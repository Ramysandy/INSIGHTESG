import os
from flask import Flask
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Load your CSV file containing sentiment data (data.csv)
data_df = pd.read_csv("data.csv")  # Update with the actual path to your CSV file

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),
    dcc.Textarea(id='input_text', placeholder="Enter text or URL"),
    html.Button("Analyze Sentiment", id='analyze_button'),
    html.Div(id='sentiment_results')
])

@app.callback(
    Output('sentiment_results', 'children'),
    Input('analyze_button', 'n_clicks'),
    [dash.dependencies.State('input_text', 'value')]
)
def update_sentiment(n_clicks, input_text):
    if input_text:
        if input_text.startswith("http"):
            # User entered a website link, fetch text content from the URL
            input_text = fetch_text_from_url(input_text)

        # Perform sentiment analysis and get sentiment scores here
        sentiment_scores = analyze_sentiment(input_text)

        # Display the sentiment scores
        return html.Div([
            html.H3("Sentiment Analysis Results:"),
            html.P(f"Positive: {sentiment_scores['Positive']}"),
            html.P(f"Negative: {sentiment_scores['Negative']}"),
            html.P(f"Neutral: {sentiment_scores['Uncertainty']}"),
            html.P(f"Litigious: {sentiment_scores['Litigious']}"),
            html.P(f"Strong Modal: {sentiment_scores['StrongModal']}"),
            html.P(f"Weak Modal: {sentiment_scores['WeakModal']}"),
            html.P(f"Overall Sentiment: {sentiment_scores['Sentiment']}"),
        ])

    return ""

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = " ".join([p.text for p in soup.find_all('p')])  # Extract text from <p> tags
            return text
        else:
            return "Error: Unable to fetch text from the provided URL."
    except Exception as e:
        return "Error: " + str(e)

def analyze_sentiment(text):
    # Use VADER sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Define your custom sentiment categories
    sentiment_categories = {
        "Negative": sentiment_scores["neg"],
        "Positive": sentiment_scores["pos"],
        "Uncertainty": sentiment_scores["neu"],  # Neutral sentiment
        "Litigious": 0.0,  # Define your own logic for this category
        "StrongModal": 0.0,  # Define your own logic for this category
        "WeakModal": 0.0,  # Define your own logic for this category
    }

    # You can set your own thresholds and logic for the custom categories
    # Example: Classify as Litigious if compound_score is greater than 0.5
    if sentiment_scores["compound"] > 0.5:
        sentiment_categories["Litigious"] = 1.0

    # Example: Classify as StrongModal if compound_score is less than -0.5
    if sentiment_scores["compound"] < -0.5:
        sentiment_categories["StrongModal"] = 1.0

    # Example: Classify as WeakModal if compound_score is between -0.5 and 0.5
    if -0.5 <= sentiment_scores["compound"] <= 0.5:
        sentiment_categories["WeakModal"] = 1.0

    # Calculate the overall Sentiment based on custom categories
    if sentiment_categories["Positive"] > sentiment_categories["Negative"]:
        sentiment = "Positive"
    elif sentiment_categories["Negative"] > sentiment_categories["Positive"]:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Add the overall Sentiment to the dictionary
    sentiment_categories["Sentiment"] = sentiment

    # Return the custom sentiment categories including overall Sentiment
    return sentiment_categories

if __name__ == "__main__":
    app.run_server(debug=True)
