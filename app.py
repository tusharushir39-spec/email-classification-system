from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

DATABASE = "email_classifier.db"


training_emails = [
    "your examination timetable is available",
    "please submit your college assignment",
    "semester results are published",
    "college lecture schedule",
    "academic notice for students",

    "meeting scheduled with the manager",
    "please attend the office meeting",
    "complete the work report",
    "official project discussion",
    "employee meeting tomorrow",

    "special discount available today",
    "buy now and get 50 percent discount",
    "limited time shopping offer",
    "free delivery on your order",
    "exclusive sale offer",

    "congratulations you won a cash prize",
    "you have won a lottery",
    "claim your free reward now",
    "you won a free gift",
    "urgent prize claim required",

    "security alert new login detected",
    "your password was changed",
    "verify your account for security",
    "new device login detected",
    "account security verification required",

    "hello how are you",
    "let us meet tomorrow",
    "happy birthday",
    "please call me when you are free",
    "hope you are doing well"
]


training_labels = [
    "Education",
    "Education",
    "Education",
    "Education",
    "Education",

    "Work",
    "Work",
    "Work",
    "Work",
    "Work",

    "Promotional",
    "Promotional",
    "Promotional",
    "Promotional",
    "Promotional",

    "Spam",
    "Spam",
    "Spam",
    "Spam",
    "Spam",

    "Security",
    "Security",
    "Security",
    "Security",
    "Security",

    "Personal",
    "Personal",
    "Personal",
    "Personal",
    "Personal"
]


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(training_emails)

model = MultinomialNB()

model.fit(
    X,
    training_labels
)


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def create_database():

    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/classify",
    methods=["POST"]
)
def classify():

    email = request.form.get(
        "email",
        ""
    ).strip()

    if not email:

        return render_template(
            "index.html",
            error="Please enter an email message."
        )

    cleaned_email = clean_text(email)

    features = vectorizer.transform(
        [cleaned_email]
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = max(
        probabilities
    ) * 100

    confidence = round(
        confidence,
        2
    )


    connection = sqlite3.connect(
        DATABASE
    )

    connection.execute(
        """
        INSERT INTO history
        (email, category, confidence)
        VALUES (?, ?, ?)
        """,
        (
            email,
            prediction,
            confidence
        )
    )

    connection.commit()

    connection.close()


    return render_template(
        "result.html",
        email=email,
        category=prediction,
        confidence=confidence
    )


@app.route("/history")
def history():

    connection = sqlite3.connect(
        DATABASE
    )

    records = connection.execute(
        """
        SELECT
            id,
            email,
            category,
            confidence,
            created_at
        FROM history
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.execute(
        "DELETE FROM history"
    )

    connection.commit()

    connection.close()

    return redirect(
        url_for("history")
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


if __name__ == "__main__":

    create_database()

    app.run(
        debug=True
    )