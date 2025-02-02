from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class BookRecommender:
    def __init__(self):
        self.df = None
        self.similarity_matrix = None

    def load_data(self, filepath):
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {filepath}")
        except ValueError as e:
            raise ValueError(f"Error loading data: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def preprocess_data(self, df, summary_column='summary', title_column='title'):
        if df[summary_column].isnull().any():
            df[summary_column] = df[summary_column].fillna('')
            print("Handled missing values in summary column.")

        if df[title_column].isnull().any():
            df[title_column] = df[title_column].fillna('')
            print("Handled missing values in title column.")

        df = df.drop_duplicates(subset=[title_column, summary_column], keep='first')
        print("Removed duplicate rows.")

        df = df[~(df[title_column] == '') | (df[summary_column] == '')]
        print("Removed rows with blank title and summary.")

        return df

    def create_tfidf_matrix(self, df, summary_column='summary'):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df[summary_column])
        return tfidf_matrix, tfidf

    def calculate_similarity(self, tfidf_matrix):
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    def recommend_books(self, book_title):
        try:
            book_index = self.df[self.df['title'] == book_title].index[0]
        except IndexError:
            return "Book title not found."
        except Exception as e:
            return f"An error occurred: {e}"

        similar_books_indices = self.similarity_matrix[book_index].argsort()[::-1][1:6]  # Fixed top_n to 5
        recommended_books = self.df['title'].iloc[similar_books_indices].tolist()
        return recommended_books

    def load_and_process_data(self, filepath):
        try:
            self.df = self.load_data(filepath)
            self.df = self.preprocess_data(self.df)
            tfidf_matrix, _ = self.create_tfidf_matrix(self.df)
            self.similarity_matrix = self.calculate_similarity(tfidf_matrix)
            return True
        except Exception as e:
            print(f"Error during data loading/processing: {e}")
            return False


recommender = BookRecommender()

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    recommendations = None # Initialize recommendations
    if request.method == "POST":
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    filepath = "uploaded_file." + file.filename.rsplit('.', 1)[1]
                    file.save(filepath)
                    if recommender.load_and_process_data(filepath):
                        message = "File uploaded and processed successfully!"
                    else:
                        message = "Error processing the file."
                except Exception as e:
                    message = f"File upload failed: {e}"
            else:
                message = "No file selected."

        elif 'book_title' in request.form:
            book_title = request.form['book_title']
            if recommender.df is None or recommender.similarity_matrix is None:
                message = "Please upload and process a file first."
            else:
                recommendations = recommender.recommend_books(book_title)
                if isinstance(recommendations, str): # Check if it is an error message.
                    message = recommendations
                else:
                    message = "" # Clear any previous messages.
    return render_template("index.html", message=message, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)