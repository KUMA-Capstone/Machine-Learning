# Data engineering
We use two different datasets sourced from Kaggle. The first dataset is the Daylio Mood data, which contains 1000 daily moods along with activity categories. The second dataset is the Text Sentiment Data, which consists of data from Twitter. We divide our analysis based on these two datasets.

- Daylio Mood Data

  This data includes features such as full_date, date, weekday, time, sub_mood, activities, and mood. We only use the weekday, sub_mood, and activities features. Based on the weekday feature, we created a new feature called "is_weekend" to indicate whether the day is a weekend or not. If it is a weekend, it is assigned a value of 1, otherwise 0.
  
  After performing feature engineering, the next step is to split the dataset into training and testing sets. The mood feature plays the role of the label. We use a splitting ratio of 60% for training data and 40% for testing data. This ratio is chosen because our data is insufficient, and a larger testing dataset is required. The training data will increase later due to oversampling.
  
  Oversampling is performed because the distribution of each label value differs significantly. This is necessary to avoid overfitting of the model. After oversampling, encoding is applied. For the independent variables, we use one-hot encoding, while for the dependent variable, we use ordinal encoding since the label values need to be ordered from 0 (very poor) to 4 (very good).

- Text Sentiment Data

  This data consists of text and mood features. The text feature will be used as an independent variable, and mood will be the dependent variable. Processing the text data is slightly more complex compared to the Daylio data processing scheme due to its larger size, which poses challenges in terms of time and memory complexity.
  
  The processing steps for the text data include:
  
  - Removing HTML syntax.

  - Removing special and punctuation characters.

  - Removing links.

  - Removing stopwords.
  
  - Removing non-ASCII characters.

  - Converting text to lowercase.

  - Tokenizing the data.
 
# Model Deployment <br>
We deploy the model using Flask by creating a Python code that will load the Keras h5 model. The Flask will receive a body request from the web server that is sent by the mobile app as a json from the backend. Then we preprocess the data we got from the body request, and we preprocess so the data shape becomes similar to the data train we used to build our model. After that, the model will predict the child mood of the user for the given body request. The recommendation is returned in JSON format.

# Machine Learning Model <br>
Considering the context and the available data, we have selected several models, including:

- Artificial Neural Network (ANN)

  ANN is used for the Daylio data because, after preprocessing, the final output is categorical. We use multiple layers that receive input information from other neurons. Neurons in the ANN process the information. We achieved an accuracy of 98% on the validation data.

- BERT
  
  Once the text data is prepared, it is fed into the BERT model. BERT has been pre-trained on a massive amount of unlabeled text data, which enables it to learn contextual representations of words. During the fine-tuning phase, the BERT model is trained on our specific task, which is sentiment analysis in this case. By fine-tuning BERT on our labeled data, it learns to associate the textual patterns with the corresponding mood labels.

  The output of the BERT model is then used for sentiment analysis predictions. It can classify the sentiment of a given text into different categories, such as positive, negative, or neutral. The performance of the BERT model is evaluated based on metrics like accuracy, precision, recall, and F1 score.

- Ensemble Weighted Average

  Ensemble weighted average is a technique used to combine predictions from multiple models. By utilizing weighted averaging, a final result is obtained. We used a weight of 0.7 for the NLP/BERT model and a weight of 0.3 for the ANN model.
