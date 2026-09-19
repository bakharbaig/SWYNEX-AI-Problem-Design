# SWYNEX-AI-Problem-Design

## Project Title
**Student Feedback Sentiment Classifier**

## 1. Practical AI Problem

Colleges receive many student feedback comments. Reading and manually categorizing every comment can take time. This project uses a simple Natural Language Processing (NLP) model to classify student feedback into three categories:

- Positive
- Negative
- Neutral

The goal is to demonstrate a small, practical AI classification problem using a manageable text dataset.

## 2. Target User

The primary users are:
- College faculty
- Academic coordinators
- Students/feedback administrators

## 3. Data Source

For this internship task, a small manually created sample dataset of student feedback is included in `dataset.csv`.

The dataset contains two columns:

- `feedback` — the student's text feedback
- `sentiment` — the expected category

No personal information is included.

## 4. AI Approach

The project uses:

1. **TF-IDF Vectorization** — converts text into numerical features.
2. **Logistic Regression** — classifies the feedback into Positive, Negative, or Neutral.

The dataset is divided into training and testing data. The test set is used to evaluate the model.

## 5. Constraints

- Small demonstration dataset
- English text only
- No personal or sensitive student information
- The model is intended as a prototype, not a production college feedback system
- Accuracy can change when the model is trained on a larger or more diverse dataset

## 6. Evaluation Approach

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The program prints these results after training.

## 7. Success Criteria

The project is considered successful if:

- The model can classify new feedback into the three defined categories.
- The prototype reaches approximately **80% or higher test accuracy** on a suitable test split.
- The results are displayed clearly.
- The solution can be run on a normal laptop using Python.

Because this is a small demonstration dataset, the result should not be treated as evidence of real-world performance.

## 8. How to Run

### Step 1: Install Python
Install Python 3.9 or newer.

### Step 2: Open the project folder
Open Command Prompt/Terminal inside this folder.

### Step 3: Install libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python sentiment_classifier.py
```

### Step 5: Try your own feedback

When the program asks for feedback, type a sentence such as:

```text
The practical classes were very helpful.
```

The model will display the predicted sentiment.

Type `exit` to stop the program.

## 9. Files

```text
SWYNEX-AI-Problem-Design/
│
├── dataset.csv
├── sentiment_classifier.py
├── requirements.txt
├── README.md
└── linkedin_video_script.txt
```

## 10. Future Improvements

- Use a much larger real-world dataset.
- Add more feedback categories.
- Support multiple Indian languages.
- Build a web interface.
- Compare multiple machine-learning algorithms.
- Add a dashboard showing feedback trends.

## 11. Ethical Considerations

Student feedback should be handled responsibly. A real deployment should protect student privacy, remove personally identifiable information, and avoid making important decisions solely from an automated prediction.
