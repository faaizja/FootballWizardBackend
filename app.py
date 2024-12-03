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
        
        # Log the received prompt
        print(f"Received prompt from frontend: {user_input}")

        # Get relevant data based on the query
        relevant_data, context = get_relevant_data(user_input)

        if relevant_data:
            system_message = f"""You are a Premier League stat analyst. Use the following {context} data to answer the question:
            {relevant_data}
            
            Provide a concise and accurate answer based solely on the data provided. If the data doesn't contain the exact information needed, use the closest relevant information and explain any assumptions or limitations."""
        else:
            system_message = "You are a Premier League stat analyst. If asked about recent statistics, inform the user that you don't have access to the most recent data."

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",    
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ]
        )

        ai_response = completion.choices[0].message['content']

        # Store conversation in database
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