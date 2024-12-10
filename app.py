from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import openai
from dotenv import load_dotenv
import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up logging 
logging.basicConfig(level=logging.INFO)

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CSV files
goal_stats_df = pd.read_csv('goal_stats.csv')
assist_stats_df = pd.read_csv('assists_stats.csv')
clean_sheet_stats_df = pd.read_csv('defender_clean_sheets_stats.csv')

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="football_wizard",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        print("Connected to database")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Unable to connect to the database: {e}")
        return None
    
documents_date = [
    "2019", "2020", "2021", "2022", "2023", "2024",
    "2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024"
]

embeddings_dates = embedding_model.encode(documents_date)

df_dates = pd.DataFrame({"Document": documents_date, "Embedding": list(embeddings_dates)})

def retrieve_with_pandas_date(query, top_k=1):
    query_embedding = embedding_model.encode([query])[0]
    df_dates['Similarity'] = df_dates['Embedding'].apply(lambda x: np.dot(query_embedding, x) / (np.linalg.norm(query_embedding) * np.linalg.norm(x)))
    filtered_results = df_dates[df_dates['Similarity'] > 0.3].sort_values(by="Similarity", ascending=False).head(top_k)

    if filtered_results.empty:
        return False
    else:
        return True
    
def without_data(query):
    system_message = "You are a Premier League statistics analyst. Answer the following question accurately, concisely, and with a focus on relevant data."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    
    return completion

def with_data(query, system_message):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    
    return completion
    

def get_relevant_data(query):
    # Determine which CSV file to use based on the query
    if 'goal' in query.lower():
        df = goal_stats_df
        context = "goal scorers"
    elif 'assist' in query.lower():
        df = assist_stats_df
        context = "assist providers"
    elif 'clean sheet' in query.lower():
        df = clean_sheet_stats_df
        context = "clean sheet statistics"
    else:
        return None, None

    # Extract year from query
    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        year = year_match.group(1)
        df_filtered = df[df['Year'].str.contains(year)]
    else:
        df_filtered = df

    if df_filtered.empty:
        return None, None

    # Convert dataframe to string
    data_string = df_filtered.to_string(index=False)

    return data_string, context

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data['message']
        
        relevant_data = None
        context = None
        
        # Log the received prompt
        print(f"Received prompt from frontend: {user_input}")

        if retrieve_with_pandas_date(user_input):
            relevant_data, context = get_relevant_data(user_input)

        if relevant_data:
            system_message = f"""You are a Premier League stat analyst. Use the following {context} data to answer the question: {relevant_data} Provide a concise and accurate answer based solely on the data provided. If the data doesn't contain the exact information needed example the year 2019 or prior use any information from the internet or anywhere else to answer the question, use the closest relevant information and explain any assumptions or limitations."""
            completion = with_data(system_message, user_input)
        else:
            completion = without_data(user_input)

        ai_response = completion.choices[0].message['content']

        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("INSERT INTO conversations (user_input, ai_response) VALUES (%s, %s)",
                            (user_input, ai_response))
                conn.commit()
                print("Successfully inserted into database")
            except Exception as e:
                logging.error(f"Database error: {e}")
            finally:
                cur.close()
                conn.close()
        else:
            logging.error("Failed to connect to the database")

        return jsonify({"response": ai_response})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred processing your request"}), 500

if __name__ == '__main__':
    app.run(debug=True)