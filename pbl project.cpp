import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pytesseract as tess
from PIL import Image
import difflib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    if not image_path:
        return "No image file provided"

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return "Image file not found"
    text = tess.image_to_string(img)
    return text

# Function to predict the number of books borrowed
def predict_books_borrowed(data_path):
    if not data_path:
        return "No data file provided"

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        return "Data file not found"

    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Extract features from the date column
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday

    # Prepare the data for training
    X = data[['year', 'month', 'day', 'weekday']]
    y = data['books_borrowed']

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the current date and extract features
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day
    current_weekday = current_date.weekday()

    # Create a new DataFrame with the current date's features
    new_data = pd.DataFrame({'year': [current_year],
                             'month': [current_month],
                             'day': [current_day],
                             'weekday': [current_weekday]})

    # Predict the number of books to be borrowed
    prediction = model.predict(new_data)

    return prediction[0]

# Function to suggest similar books based on user input
def suggest_similar_books(books, book_name):
    # Fill missing values in the DataFrame
    books.fillna('', inplace=True)

    selected_features = ['title', 'author', 'publisher', 'genres']

    combined_features = ''

    for feature in selected_features:
        if feature in books.columns:
            combined_features += books[feature] + ' '

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vectors)

    list_of_all_titles = books['title'].tolist()

    find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_book = books[books['title'] == close_match].index[0]

        similarity_score = list(enumerate(similarity[index_of_the_book]))

        sorted_similar_books = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        return sorted_similar_books[:30]
    else:
        return []

# Function to detect library card fraud using logistic regression
def detect_fraud(library_cards_data_path, card_number):
    library_cards = pd.read_csv(library_cards_data_path)

    # Prepare the data for training
    X = library_cards[['card_number']]
    y = library_cards['fraud']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Predict fraud for the given card number
    prediction = model.predict([card_number])

    return prediction[0], accuracy

def main():
    st.title("Library Management System")

    menu_choice = st.sidebar.selectbox("Select an option:", ["Book Borrow Prediction", "Text Extraction from Image", "Book Suggestions", "Library Card Fraud Detection"])

    if menu_choice == "Book Borrow Prediction":
        st.header("Book Borrow Prediction")

        data_path = st.text_input("Enter the path to historical data CSV file:")
        prediction = predict_books_borrowed(data_path)

        st.subheader("Predicted Number of Books to be Borrowed:")
        st.write(prediction)

    elif menu_choice == "Text Extraction from Image":
        st.header("Text Extraction from Image")

        image_path = st.text_input("Enter the path to the image file:")
        extracted_text = extract_text_from_image(image_path)

        st.subheader("Extracted Text:")
        st.write(extracted_text)

    elif menu_choice == "Book Suggestions":
        st.header("Book Suggestions")

        books_data_path = st.text_input("Enter the path to books data CSV file:")
        book_name = st.text_input("Enter the name of the book you're interested in:")
        
        if st.button("Find Similar Books"):
            books = pd.read_csv(books_data_path)
            similar_books = suggest_similar_books(books, book_name)

            st.subheader("Suggested Similar Books:")
            for i, book in enumerate(similar_books):
                index = book[0]
                title_from_index = books.loc[index, 'title']
                title_from = books.loc[index, 'publisher']
                if i < 30:
                    st.write(f"{i + 1} - {title_from_index} from Publication {title_from}")

    elif menu_choice == "Library Card Fraud Detection":
        st.header("Library Card Fraud Detection")

        library_cards_data_path = st.text_input("Enter the path to library cards data CSV file:")
        card_number = st.text_input("Enter the library card number:")

        if st.button("Detect Fraud"):
            fraud_prediction, accuracy = detect_fraud(library_cards_data_path, card_number)

            st.subheader("Fraud Prediction:")
            if fraud_prediction == 0:
                st.write("Not Fraudulent")
            else:
                st.write("Fraudulent")

            st.subheader("Accuracy:")
            st.write(accuracy)

if _name_ == "_main_":
    main()