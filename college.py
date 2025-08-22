# college_feedback_analysis.py
# Internship Project: College Event Feedback Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk

# Download VADER if needed
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ------------------ Load Data ------------------
# Replace with your Google Forms CSV file
file_path = "event_feedback.csv"
df = pd.read_csv(file_path)

print("Sample Data:")
print(df.head())

# ------------------ Data Cleaning ------------------
# Rename columns (adjust according to your form structure)
df.rename(columns={
    "How would you rate the event?": "Rating",
    "What did you like about the event?": "Positive_Feedback",
    "What can be improved?": "Improvement_Feedback"
}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# ------------------ Sentiment Analysis ------------------
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not isinstance(text, str): 
        return "Neutral"
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Positive_Sentiment"] = df["Positive_Feedback"].apply(get_sentiment)
df["Improvement_Sentiment"] = df["Improvement_Feedback"].apply(get_sentiment)

# ------------------ Insights ------------------
print("\nAverage Rating:", df["Rating"].mean())
print("Sentiment Counts:\n", df["Positive_Sentiment"].value_counts())

# ------------------ Charts ------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Rating", data=df, palette="viridis")
plt.title("Event Rating Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Positive_Sentiment", data=df, palette="Set2")
plt.title("Positive Feedback Sentiment")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Improvement_Sentiment", data=df, palette="Set1")
plt.title("Improvement Feedback Sentiment")
plt.show()

# ------------------ Word Clouds (Optional) ------------------
from wordcloud import WordCloud

positive_text = " ".join(str(x) for x in df["Positive_Feedback"])
improvement_text = " ".join(str(x) for x in df["Improvement_Feedback"])

plt.figure(figsize=(8,4))
wc1 = WordCloud(width=400, height=300, background_color="white").generate(positive_text)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title("What Students Liked")
plt.show()

plt.figure(figsize=(8,4))
wc2 = WordCloud(width=400, height=300, background_color="white").generate(improvement_text)
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title("Areas for Improvement")
plt.show()
