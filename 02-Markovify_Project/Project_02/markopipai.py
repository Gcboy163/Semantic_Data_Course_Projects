import ast
import pandas as pd
import markovify

df = pd.read_csv('RAW_recipes.csv')

# Create a new dataframe with two columns
df1 = df[['name', 'steps']].copy()

# Remove missing values (NaN) for recipes with no description
df1 = df1.dropna(subset=['steps'])

# Choose a larger number of rows for modeling
n = 1000

# Process the data to tokenize sentences
data_subset = df1['steps'].iloc[0:n].apply(ast.literal_eval)
#print("\nprinting data subset:\n")
#print(data_subset)
#print("\n")

# Create a list of sentences
sentences = [sentence for sublist in data_subset for sentence in sublist]
#print("printing sentences:\n")
#print(sentences)

# Create a Markovify model # A state size of 2 means it looks at the previous two words to predict the next word.
text_model = markovify.Text(sentences, state_size=3)

# Generate text
for _ in range(10):
    print("\n")
    print(text_model.make_sentence())
