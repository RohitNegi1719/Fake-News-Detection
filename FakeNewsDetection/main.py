import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

# Step 1: Load the dataset
df = pd.read_csv(r'C:\Users\Rohit Negi\Desktop\FakeNewsDetection\IFND.csv', encoding='iso-8859-1')

# Step 2: Preprocess the data
df['Fake'] = df['Label'].apply(lambda x: 0 if x == "TRUE" else 1)
df = df.drop(['id', 'Image', 'Web', 'Category', 'Date', 'Label'], axis=1)

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Removing stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    #tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

df['Statement'] = df['Statement'].apply(preprocess_text)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Statement'], df['Fake'], test_size=0.2, random_state=42)

# Step 4: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Step 5: Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(tfidf_train, y_train)

# Step 6: Predictions
y_pred = naive_bayes_classifier.predict(tfidf_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
