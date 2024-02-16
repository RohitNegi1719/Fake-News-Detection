# Fake News Detection

## Overview
This project implements a Fake News Detection system using a Multinomial Naive Bayes classifier trained on TF-IDF features. The system preprocesses text data, splits it into training and testing sets, extracts features using TF-IDF, trains the classifier, makes predictions, and evaluates the model's performance.

## Dataset
The dataset used for this project is `IFND.csv`, which contains statements labeled as "TRUE" or "FAKE" along with other attributes. The data is preprocessed to convert the label into binary (0 for TRUE, 1 for FAKE) and to clean and preprocess the text for analysis.

## Dependencies
- pandas
- scikit-learn
- nltk

## Usage
1. Ensure you have the dataset `IFND.csv` available.
2. Run the script `fake_news_detection.py`.
3. The script will preprocess the data, train the Multinomial Naive Bayes classifier, make predictions on the test data, and evaluate the model's accuracy.
4. The accuracy and classification report will be printed to the console.

## Results
After running the script, you'll get an accuracy score and a detailed classification report showing precision, recall, and F1-score for each class (TRUE and FAKE).

## Contributing
Contributions to improve the code, add new features, or fix issues are welcome! Please feel free to submit pull requests or open issues.

## Authors
Rohit Negi

## Acknowledgments
- Thanks to the authors of the dataset used in this project.

