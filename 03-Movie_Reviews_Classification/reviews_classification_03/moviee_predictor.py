# import all requiredd libraries
import pandas as pd #Data manipulation
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

# load dataset
print("\nloading dataset... \n")
df = pd.read_csv('movies_dataset.txt', sep = ' ::: ', header = None, engine = 'python')

# display some headings
print("\nDisplay some dataset's headings: \n")
print(df.head())

# remove some information and build another dataframe that contains "Genre" and "Description" 
# predict movie genre based on the movie description
# Create a new dataframe with from column 2 ('Genre') and 3 ('Description')
print("\nCreating new dataframe from copying column 2 ('Genre') and 3 ('Description')...\n")
df1 = df[[2,3]].copy()

# Remove missing values (NaN) from  the  copied columns
df1 = df1[pd.notnull(df1[2])]
df1 = df1[pd.notnull(df1[3])]

# Renaming columns for a simpler name
df1.columns = ['Genre', 'Description']

# display new dataset
print("\nNew dataframe with renamed headings: \n")
print(df1.head())

# Checking percentage of Description that had text field
print("\nChecking percentage of Description that had text field: ")
total = df1['Description'].notnull().sum()
print(round((total/len(df)*100),1))

# Classes or categories of Genre
print("\nTake a glance at Genre's Categories/Classes: \n")
print(pd.DataFrame(df1.Genre.unique()).values)
print('\nTotal Categories: ')
print(df1.Genre.unique().shape)

# check dataset dimension
print("\nDataset's dimension: ")
print(df1.shape)

# let's work with a smaller sample of the data to speed things up
print("\nTake small sample to speed things up... \n")
df2 = df1.sample(20000, random_state=1).copy()

# check  small dataset dimension
print("Small dataset's dimension: ")
print(df2.shape)

#  Assigning each Classes/Categories  to values for easy processing
print("\nEnumerate Classes/Categories  to values for easy processing...\n")
pd.DataFrame(df2.Genre.unique())

# Display total values for each categories
print("\nDisplaying total values for each categories: \n")
print(df2.Genre.value_counts())

'''# Plot graph of the whole dataframe
print("\nPlotting graph for the whole sample: \n")
fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey', 'grey','darkblue','darkblue','darkblue']
df1.groupby('Genre').Description.count().sort_values().plot.barh(ylim=0, color=colors, title= 'NUMBER OF MOVIES FOR EACH GENRE\n')
plt.xlabel('Number of ocurrences', fontsize = 10);
draw() '''

# Plot graph of the small dataframe
print("\nPlotting graph for the small sample: \n")
fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey', 'grey','grey','grey','grey','grey',
                'grey','grey','grey','grey','grey', 'grey','grey','grey','grey','grey',
                'grey', 'grey','grey','grey','darkgreen','darkgreen','darkgreen']
df2.groupby('Genre').Description.count().sort_values().plot.barh(ylim=0, color=colors, title= 'NUMBER OF MOVIES FOR EACH GENRE (SMALL SAMPLE)\n')
plt.xlabel('Number of ocurrences', fontsize = 10);
draw()

# Create a new column 'Genre_id' with encoded categories
print("\nCreating new Genre_id column with assigned encoding: \n")
df2['Genre_id'] = df2['Genre'].factorize()[0]
genre_id_df = df2[['Genre', 'Genre_id']].drop_duplicates()

# Dictionaries for future use
genre_to_id = dict(genre_id_df.values)
id_to_genre = dict(genre_id_df[['Genre_id', 'Genre']].values)

# New dataframe
print(df2.head())

# Calculate TF-IDF Scores
print("\nCalculating TF-IDF Scores: ")

# min_df=5: Ignore terms that have a document frequency strictly lower than 5. These terms are too rare to be meaningful.
#  ngram_range=(1, 2): Consider unigrams (single words) and bigrams (pairs of adjacent words) for analysis.
#  stop_words='english': Ignore common English words ('and', 'the', etc.) that typically don't contain much information.
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

# We transform each Description into a vector matrix
features = tfidf.fit_transform(df2.Description).toarray()

# assigns the genre ids from the DataFrame df2 to the variable labels, which likely will be used as target labels for a machine learning model.
labels = df2.Genre_id

# Display the size of the feature matrix
# The features.shape gives a tuple (number of descriptions, number of features), indicating the number of descriptions and how many features (unique unigrams and bigrams) represent each description after the transformation.
print("Each of the %d descriptions is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

# Finding the three most correlated terms with each of the Genre categories
print("\nTop 3 most correlated terms with each Genre's textual data: ")
N = 3
for Genre, Genre_id in sorted(genre_to_id.items()):
  features_chi2 = chi2(features, labels == Genre_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(Genre))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

# classification: train-test, split
print("\nStarting data classification...: split dataset to TRAINING and TESTING...")
X = df2['Description'] # Collection of documents
y = df2['Genre'] # Target or the labels we want to predict (i.e., the 27 different movie genres)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 20)

#  train several models, Random Forest, Linear Support Vector Machine, Multinomial Naive Bayes, Logistic Regression
print("\nTraining different models: Random Forest, Linear Support Vector Machine, Multinomial Naive Bayes, Logistic Regression\n")
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(dual=True),
    MultinomialNB(),
    LogisticRegression(random_state=0, max_iter=400),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

# too lazy to try
'''scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)'''

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']

print("\nAccuracy comparison of each model: \n")
print(acc)

# plot accuracy graph for  each model
print("\nAccuracy graph for each model: \n")
plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy',
            data=cv_df,
            color='lightblue',
            showmeans=True)
plt.title("MEAN ACCURACY (cv = 5)\n", size=14);
draw()

# evaluating the model
print("\nEvaluating the model using test dataset... \n")
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features,
                                                               labels,
                                                               df2.index, test_size=0.25,
                                                               random_state=1)
model = LinearSVC(dual=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
print('\t\t\t\tCLASSIFICATION METRICS REPORT\n')
print(metrics.classification_report(y_test, y_pred, target_names=df2['Genre'].unique(), zero_division=0))

#  Confusion matrix
print('\nPlotting Confusion matrix graph...\n')
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=genre_id_df.Genre.values,
            yticklabels=genre_id_df.Genre.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16);
draw()

# Analysing Results
print('\nAnalysing Results: Let’s have a look at the texts that were wrongly classified')
for predicted in genre_id_df.Genre_id:
  for actual in genre_id_df.Genre_id:
    if predicted != actual and conf_mat[actual, predicted] >= 20:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_genre[actual],
                                                           id_to_genre[predicted],
                                                           conf_mat[actual, predicted]))

      display(df2.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Genre', 'Description']])
      print('')

# Most correlated terms for each category
print('\nMost correlated terms for each category¶\n')
model.fit(features, labels)

N = 4
for Genre, Genre_id in sorted(genre_to_id.items()):
  indices = np.argsort(model.coef_[Genre_id])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("\n==> '{}':".format(Genre))
  print("  * Top unigrams: %s" %(', '.join(unigrams)))
  print("  * Top bigrams: %s" %(', '.join(bigrams)))

# Calling the model for predicting new genre based on new movie description
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)

# Genre: Comedy, Romance, School, Shounen
print('\nLet the model predicts the genre of Nozaki-kun anime: ')
new_description = """High school student Sakura Chiyo has a crush on schoolmate Nozaki Umetarou, but when she confesses her love to him, he mistakes her for a fan and gives her an autograph. When she says that she always wants to be with him, he invites her to his house, but has her help on some drawings. Chiyo discovers that Umetarou is actually a renowned shoujo manga artist named Yumeno Sakiko, but agrees to be his assistant. As they work on his manga Let's Love, they encounter other schoolmates who assist them or serve as inspirations for characters in the stories."""
print(model.predict(fitted_vectorizer.transform([new_description])))

#  command to keep all figures open for further inspectsyen....
show()

