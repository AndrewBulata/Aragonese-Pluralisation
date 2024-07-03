# Aragonese-Pluralisation
---
A computational answer to a problem by Bruno L’Astorina for the 2016 Brazilian Linguistics Olympiad 

### Introduction to Aragonese and its Linguistic Features

![Linguistic_map_Southwestern_Europe-en](https://github.com/AndrewBulata/Aragonese-Pluralisation/assets/64040990/3dd0ea6a-273b-4565-8b59-3c897e7850ac)

                              Map designed by Alexandre Vigo at Galician Wikipedia.

Aragonese is a Romance language spoken primarily in the autonomous community of Aragon in northeastern Spain. Despite its relatively small number of speakers, ranging between 10,000 and 30,000, Aragonese holds historical and cultural significance. It evolved from Vulgar Latin, like other Romance languages such as Spanish, Catalan, and Portuguese, but has retained unique phonetic, grammatical, and lexical characteristics due to its geographical and historical context. Aragonese is recognised for preserving some archaic features lost in neighbouring languages and incorporating influences from Basque, Catalan, and Occitan.

One interesting aspect of Aragonese grammar is the formation of plurals. Similar to other Romance languages, Aragonese typically forms plurals by adding an -s to the end of singular nouns. For instance, "bal" (valley) becomes "bals" and "cantal" (stone) becomes "cantals". However, there are exceptions and irregular plural forms that must be memorised, such as "banquet" to "banquetz" and "clot" to "clotz". Understanding these pluralisation rules provides insight into the language's structure and helps in deciphering texts and linguistic problems.

### The Aragonese Problem

The document includes a specific problem set from the Brazilian Linguistics Olympiad 2016, which focuses on the formation of plural nouns in Aragonese. The problem presents a list of Aragonese words, their plural forms, and their translations, with some plural forms missing. Here are the provided words and their translations, with the task being to fill in the missing plurals:

| Translation   | Singular     | Plural       |
|------------|------------|-------------------|
| valley     | bal        | bals              |
| stool      | banquet    | banquetz          |
| hole       | clot       | clotz             |
| stone      | cantal     | cantals           |
| awake      | concordau  |                   |
| chocolate  | chicolat   |                   |
| union      | chunta     |                   |
| unhanded   | deixau     | deixaus           |
| eclipse    | eclix      |                   |
| cicada     | ferfet     |                   |
| character  | personache | personaches       |
| fish       | peix       | peixes            |

Do attempt the problem first and then run the script attached in this repository.

### Justification for a Machine Learning Approach to the Aragonese Pluralisation Problem

Having solved similar linguistic problems in the past, I became fascinated with the cognitive mechanisms that enable us to figure out such issues without extensive training. The human mind, with its unique 'neural network,' is the best device for language acquisition, capable of learning any language naturally over time. This curiosity led me to explore framing linguistic problems computationally. Can we develop a machine learning algorithm that learns language rules without relying on rule-based tricks? And can such a model be generalized to all languages? Stay tuned.

In addressing the Aragonese pluralisation problem, a machine learning approach is particularly effective due to the complex and irregular nature of linguistic patterns. Traditional rule-based methods might struggle to capture the nuances and exceptions in plural formation, whereas a machine learning model can learn from examples and generalise to unseen data. The chosen method utilises a Random Forest Classifier within a pipeline that incorporates feature extraction and vectorisation to capture essential characteristics of the words. Another reason why this approach is ideal for our problem is that, to replicate the challenge, we train the model on a limited dataset, requiring it to infer the rules dynamically.

#### Implementation Details

1. **Feature Extraction**: By extracting features such as the last letters of the word, its length, and counts of vowels and consonants, we provide the model with relevant linguistic patterns that influence pluralisation in Aragonese.
2. **Random Forest Classifier**: This robust classifier is selected for its ability to handle a high-dimensional feature space and its resistance to overfitting, making it suitable for capturing complex relationships within the data.
3. **Pipeline**: Using a pipeline ensures that feature extraction, vectorisation, and classification are streamlined, improving the reproducibility and efficiency of the model training and prediction processes.
4. **Training and Testing**: The model is trained on a diverse dataset of singular-plural pairs and tested on new singular nouns to validate its performance. This approach ensures the model's predictions are accurate and reliable.

This machine learning approach not only provides a scalable solution to the pluralisation problem but also demonstrates the power of data-driven methods in linguistic analysis. Below you will find the implemenation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Function to read word pairs from a text file
def read_word_pairs(file_path):
    word_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            singular, plural = line.strip().split(',')
            word_pairs.append((singular, plural))
    return word_pairs

# Define function to prepare and train the model
def train_pluralisation_model(word_pairs):
    df = pd.DataFrame(word_pairs, columns=['singular', 'plural'])

    # Extracting features for more precise pattern learning
    def extract_features(word):
        return {
            'last_letter': word[-1],
            'last_two_letters': word[-2:],
            'last_three_letters': word[-3:],
            'length': len(word)
        }

    df['features'] = df['singular'].apply(extract_features)
    df['suffix'] = df.apply(lambda row: row['plural'][len(row['singular']):], axis=1)

    X = df['features'].tolist()
    y = df['suffix']

    vectoriser = DictVectorizer(sparse=False)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=23) # a bit of brute force with 1000 est.

    pipeline = Pipeline([
        ('vectoriser', vectoriser),
        ('classifier', classifier)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Function to predict plural forms
def predict_plural(model, singular_word):
    def extract_features(word):
        return {
            'last_letter': word[-1],
            'last_two_letters': word[-2:],
            'last_three_letters': word[-3:],
            'length': len(word)
        }

    features = extract_features(singular_word)
    predicted_suffix = model.predict([features])[0]
    return singular_word + predicted_suffix

# Read the word pairs from the text file
word_pairs = read_word_pairs('aragonese_word_pairs.txt')

# Train the model
model = train_pluralisation_model(word_pairs)

# Testing the function with new singular nouns
test_words = ['concordau', 'chicolat', 'chunta', 'eclix', 'ferfet', 'bal', 'banquet', 'clot', 'banvanau', 'lau', 'crau', 'glet', 'felix']
predicted_plurals = [predict_plural(model, word) for word in test_words]

# Print the results
for singular, plural in zip(test_words, predicted_plurals):
    print(f"{singular} - {plural}")
    print('') # inserts a space between word pairs for better readability
```
The expected output should be:
```
concordau - concordaus

chicolat - chicolatz

chunta - chuntas

eclix - eclixes

ferfet - ferfetz

bal - bals

banquet - banquetz

clot - clotz

banvanau - banvanaus

lau - laus

crau - craus

glet - gletz

felix - felixes

```

Note: You might have noticed the inclusion of gibberish words such as 'banavanau', 'lau', and 'glet'. These are used to test the model's behavior when encountering new words that follow the *phonotactic* rules of Aragonese. (Phonotactics is a branch of phonology that deals with the permissible combinations of sounds in a particular language. For example, it explains why the famous phrase “Strč prst skrz krk” is Czech and not English. 🤣)

## Executing the program

In this repository, you will find the file 'Aragonese Plural Finder 90% Machine Learning.ipynb' (the same script as above) which reads the file 'aragonese_word_pairs.txt,' also included here. Make sure to download the text file in order to run the script. Please feel free to test some Aragonese words of your own, as well.

---
## Final remarks: extension to other languages


I mentioned the program is 90% ML because it still follows a pre-defined rule of examining the roots of the words. However, not all languages work this way. For example, some languages, particularly those in the Semitic family, form plurals that do not depend solely on the endings of the singular nouns. These languages often use internal changes, prefixes, suffixes, or a combination of these methods to form plurals. 

### Arabic
Arabic uses a complex system called "broken plurals" where the internal structure of the word changes, rather than simply adding a suffix. This can involve changes to the vowels within the word and sometimes to the consonants as well.
- **Kitab** (book) → **Kutub** (books)
- **Maktab** (office) → **Makatib** (offices)
- **Jabal** (mountain) → **Jibal** (mountains)

### Hebrew
Hebrew plurals are typically formed by adding suffixes, but the process is still complex due to vowel changes and variations based on gender.
- **Yeled** (boy) → **Yeladim** (boys)
- **Isha** (woman) → **Nashim** (women)
- **Bayit** (house) → **Batim** (houses)

English also displays this *phenomenon* and can form highly irregular plurals.

### English Examples

**Internal Changes (Vowel Changes)**
1. **Foot** → **Feet**
2. **Tooth** → **Teeth**
3. **Goose** → **Geese**
4. **Man** → **Men**
5. **Woman** → **Women**

**Irregular Plurals**
1. **Child** → **Children**
2. **Mouse** → **Mice**
3. **Louse** → **Lice**
4. **Ox** → **Oxen**
5. **Person** → **People**

**Foreign Plurals**
These often retain the plural forms from their languages of origin:
1. **Cactus** → **Cacti**
2. **Focus** → **Foci**
3. **Thesis** → **Theses**
4. **Crisis** → **Crises**
5. **Phenomenon** → **Phenomena**

Lastly, the current program cannot handle Italian or Romanian either because these languages rely on *vowel mutation* to form plurals, in addition to unpredictable internal changes in Romanian.

Italian: bambino - bambini (child - children)
         finestra - finestre (window - windows)
Romanian: om - oameni (human - humans)
          soră - surori (sister - sisters)

Essentially, this program is only suitable for languages that use *suffixation* (the addition of suffixes) to form plurals.

---
Therefore, the plural finder we have developed here would not detect these exceptions, as it searches for inflection patterns in the form of suffixes at the end of the word.

Additional Sources

1. **Omniglot - Hebrew**:
   - [Omniglot: Hebrew Language](https://www.omniglot.com/writing/hebrew.htm)
   
2. **Learn Hebrew Phrases - Grammar**:
   - [Learn Hebrew Phrases: Hebrew Grammar](https://www.learn-hebrew-phrases.com/hebrew-grammar/)

3. **HebrewPod101 - Hebrew Grammar**:
   - [HebrewPod101: Hebrew Grammar](https://www.hebrewpod101.com/hebrew-grammar/)




