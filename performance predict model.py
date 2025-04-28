import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


class StudentPerformancePredictor:
    def __init__(self, root):
        self.performance_combobox = None
        self.root = root
        self.root.title("AI-Based Student Performance Predictor")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Sample dataset (in a real app, you'd load from a file)
        self.data = pd.DataFrame(columns=['student_id', 'essay', 'feedback', 'sentiment', 'performance'])
        self.current_id = 1

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Input Tab
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Input Data")

        # Student ID
        ttk.Label(self.input_frame, text="Student ID:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.id_entry = ttk.Entry(self.input_frame)
        self.id_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        self.id_entry.insert(0, str(self.current_id))

        # Essay Text
        ttk.Label(self.input_frame, text="Student Essay:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.NW)
        self.essay_text = tk.Text(self.input_frame, width=60, height=10, wrap=tk.WORD)
        self.essay_text.grid(row=1, column=1, padx=10, pady=5, columnspan=2)

        # Feedback Text
        ttk.Label(self.input_frame, text="Teacher Feedback:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.NW)
        self.feedback_text = tk.Text(self.input_frame, width=60, height=5, wrap=tk.WORD)
        self.feedback_text.grid(row=2, column=1, padx=10, pady=5, columnspan=2)

        # Performance Level
        ttk.Label(self.input_frame, text="Performance Level:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        self.performance_var = tk.StringVar()
        self.performance_combobox = ttk.Combobox(self.input_frame, textvariable=self.performance_var,
                                                 values=["High", "Medium", "Low"])
        self.performance_combobox.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
        self.performance_combobox.current(0)

        # Buttons
        self.add_button = ttk.Button(self.input_frame, text="Add to Dataset", command=self.add_to_dataset)
        self.add_button.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)

        self.clear_button = ttk.Button(self.input_frame, text="Clear", command=self.clear_fields)
        self.clear_button.grid(row=4, column=2, padx=10, pady=10, sticky=tk.W)

        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis")

        # Train Model Button
        self.train_button = ttk.Button(self.analysis_frame, text="Train Prediction Model", command=self.train_model)
        self.train_button.grid(row=0, column=0, padx=10, pady=10)

        # Prediction Components
        ttk.Label(self.analysis_frame, text="Enter New Essay for Prediction:").grid(row=1, column=0, padx=10, pady=5,
                                                                                    sticky=tk.W)
        self.predict_essay = tk.Text(self.analysis_frame, width=60, height=10, wrap=tk.WORD)
        self.predict_essay.grid(row=2, column=0, padx=10, pady=5)

        self.predict_button = ttk.Button(self.analysis_frame, text="Predict Performance",
                                         command=self.predict_performance)
        self.predict_button.grid(row=3, column=0, padx=10, pady=10)

        self.prediction_result = ttk.Label(self.analysis_frame, text="Prediction will appear here",
                                           font=('Helvetica', 12))
        self.prediction_result.grid(row=4, column=0, padx=10, pady=10)

        # Visualization Tab
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Visualizations")

        self.visualize_button = ttk.Button(self.visualization_frame, text="Generate Visualizations",
                                           command=self.generate_visualizations)
        self.visualize_button.pack(pady=10)

        self.canvas_frame = ttk.Frame(self.visualization_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        words = nltk.word_tokenize(text)

        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        return ' '.join(words)

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def add_to_dataset(self):
        student_id = self.id_entry.get()
        essay = self.essay_text.get("1.0", tk.END).strip()
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        performance = self.performance_var.get()

        if not essay:
            messagebox.showerror("Error", "Please enter student essay")
            return

        # Preprocess text
        processed_essay = self.preprocess_text(essay)
        processed_feedback = self.preprocess_text(feedback) if feedback else ""

        # Analyze sentiment
        sentiment = self.analyze_sentiment(essay + " " + feedback)

        # Add to dataset
        new_row = {
            'student_id': student_id,
            'essay': essay,
            'processed_essay': processed_essay,
            'feedback': feedback,
            'processed_feedback': processed_feedback,
            'sentiment': sentiment,
            'performance': performance
        }

        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        self.current_id += 1
        self.id_entry.delete(0, tk.END)
        self.id_entry.insert(0, str(self.current_id))
        self.clear_fields()

        messagebox.showinfo("Success", "Student data added to dataset successfully!")

    def clear_fields(self):
        self.essay_text.delete("1.0", tk.END)
        self.feedback_text.delete("1.0", tk.END)
        self.performance_combobox.current(0)

    def train_model(self):
        if len(self.data) < 10:
            messagebox.showerror("Error", "Not enough data to train the model. Please add at least 10 samples.")
            return

        try:
            # Prepare features
            X_text = self.data['processed_essay'] + " " + self.data['processed_feedback']
            X_tfidf = self.vectorizer.fit_transform(X_text)
            X_sentiment = self.data['sentiment'].values.reshape(-1, 1)
            X = np.hstack((X_tfidf.toarray(), X_sentiment))

            # Prepare target
            y = self.data['performance']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            messagebox.showinfo("Training Complete",
                                f"Model trained successfully!\nTest Accuracy: {accuracy:.2%}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")

    def predict_performance(self):
        if len(self.data) == 0:
            messagebox.showerror("Error", "Please train the model first with some data.")
            return

        essay = self.predict_essay.get("1.0", tk.END).strip()
        if not essay:
            messagebox.showerror("Error", "Please enter an essay to predict")
            return

        try:
            # Preprocess
            processed_essay = self.preprocess_text(essay)
            sentiment = self.analyze_sentiment(essay)

            # Vectorize
            X_text = self.vectorizer.transform([processed_essay])
            X_sentiment = np.array([[sentiment]])
            X = np.hstack((X_text.toarray(), X_sentiment))

            # Predict
            prediction = self.model.predict(X)[0]
            confidence = np.max(self.model.predict_proba(X)) * 100

            self.prediction_result.config(text=f"Predicted Performance: {prediction}\nConfidence: {confidence:.1f}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

    def generate_visualizations(self):
        if len(self.data) == 0:
            messagebox.showerror("Error", "No data available for visualization")
            return

        # Clear previous visualizations
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Student Performance Analysis")

        # Performance distribution
        performance_counts = self.data['performance'].value_counts()
        ax1.bar(performance_counts.index, performance_counts.values, color=['green', 'orange', 'red'])
        ax1.set_title("Performance Level Distribution")
        ax1.set_xlabel("Performance Level")
        ax1.set_ylabel("Number of Students")

        # Sentiment vs Performance
        sentiment_by_performance = self.data.groupby('performance')['sentiment'].mean()
        ax2.bar(sentiment_by_performance.index, sentiment_by_performance.values, color=['green', 'orange', 'red'])
        ax2.set_title("Average Sentiment by Performance Level")
        ax2.set_xlabel("Performance Level")
        ax2.set_ylabel("Average Sentiment Score")
        ax2.axhline(0, color='black', linestyle='--')

        # Adjust layout
        plt.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPerformancePredictor(root)
    root.mainloop()
    
