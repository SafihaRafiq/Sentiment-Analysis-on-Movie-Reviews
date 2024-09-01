# Sentiment-Analysis-on-Movie-Reviews
This project will involve training a model to analyze movie reviews and predict whether the sentiment expressed in the review is positive or negative. The tool will use a dataset of labeled movie reviews to train a machine learning model, which can then be used to predict the sentiment of new, unseen reviews.
# Dataset
The dataset used in this project consists of 1,000 movie reviews, each labeled as either positive or negative. The dataset was generated for this project and is included in the repository as movie_reviews.csv
# Key Features
Data Preprocessing: The reviews are preprocessed to remove noise, such as punctuation and stopwords, and then tokenized.
Feature Extraction: We use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features suitable for machine learning models.
Modeling: Various machine learning algorithms, including Naive Bayes, Logistic Regression, and Support Vector Machines (SVM), are applied to classify the reviews.
Evaluation: The performance of the models is evaluated using accuracy, precision, recall, and F1 score.
# Installation
1. Clone the repository:
    git clone https://github.com/SafihaRafiq/sentiment-analysis-movie-reviews.git
2. Navigate to the project directory:
    cd sentiment-analysis-movie-reviews
3. Run the Streamlit app:
   streamlit run app.py
