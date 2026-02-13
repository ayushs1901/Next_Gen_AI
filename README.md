# 🤖 Intent Classification Chatbot (NLP + Machine Learning)

A simple AI-based Intent Classification system built using NLTK and Scikit-Learn.
This project preprocesses text using NLP techniques and trains a Logistic Regression model
to classify user intents such as greeting, weather queries, social media actions, and exit commands.

---

## 📌 Project Overview

This project demonstrates:

- Text preprocessing using NLP
- Stopword removal
- Tokenization
- Lemmatization
- TF-IDF Vectorization
- Intent classification using Logistic Regression

It is a beginner-friendly implementation of a Natural Language Processing pipeline.

---

## 🧠 Technologies Used

- Python 3
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Google Colab / Jupyter Notebook

---

## 📂 Project Structure

```
GenAI_lec_5.ipynb
README.md
```

---

## ⚙️ How It Works

### 1️⃣ Training Data

We manually define labeled training data:

Example:

```
("hello", "greet")
("what is the weather today", "weather")
("open linkedin", "social")
("bye", "exit")
```

---

### 2️⃣ Text Preprocessing

The following NLP steps are applied:

- Convert to lowercase
- Tokenization
- Remove stopwords
- Remove punctuation
- Lemmatization
- Remove non-alphabetic words

Function used:

```python
preprocess_text(documents)
```

---

### 3️⃣ Feature Extraction (TF-IDF)

We convert cleaned text into numerical vectors using:

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_data)
```

---

### 4️⃣ Model Training

We train a Logistic Regression classifier:

```python
model = LogisticRegression()
model.fit(X, labels)
```

---

### 5️⃣ Prediction

For a new user message:

```python
user_msg = "open LinkedIn"
processed = preprocess_text([user_msg])
user_vector = vectorizer.transform(processed)
prediction = model.predict(user_vector)
print("Prediction is:", prediction)
```

Example Output:

```
Prediction is: ['social']
```

---

## 🎯 Supported Intents

| Intent   | Example |
|----------|----------|
| greet    | hello, good morning |
| weather  | what is temperature today |
| social   | open linkedin |
| open_web | open twitter |
| exit     | bye, quit |

---

## 🚀 How To Run

### Step 1: Install Dependencies

```bash
pip install nltk scikit-learn
```

### Step 2: Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 3: Run Notebook

Open in:

- Jupyter Notebook  
OR  
- Google Colab  

---

## 🧪 Example Input & Output

Input:
```
open Instagram
```

Output:
```
social
```

---

## 📈 Future Improvements

- Add train/test split and accuracy score
- Add more training data
- Convert into interactive chatbot loop
- Add confidence score
- Deploy using Flask or Streamlit
- Add deep learning (LSTM / Transformer)

---

## 🎓 Learning Outcomes

This project helps in understanding:

- NLP preprocessing pipeline
- Feature engineering for text
- Supervised machine learning
- Intent classification logic

---

## 👨‍💻 Author

Ayush Shrivastav  
B.Tech Student  
AI & Machine Learning Enthusiast  

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub!
