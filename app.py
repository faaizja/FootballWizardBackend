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
import gunicorn

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

previous_query = None
previous_response = None

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
    system_message = "You are a Premier League statistics analyst. Answer the following question accurately, concisely, and with a focus on relevant data. If the question requires additional assumptions or context, clearly explain them in your response."

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    
    return completion

def with_data(query, system_message):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    
    return completion
    

def get_relevant_data(query):
    prediction = False
    
    if 'goal' in query.lower():
        df = goal_stats_df
        context = "goal scorers"
    elif 'assist' in query.lower():
        df = assist_stats_df
        context = "assist providers"
    elif 'clean sheet' in query.lower():
        df = clean_sheet_stats_df
        context = "clean sheet statistics"
    elif 'clean sheets' in query.lower():
        df = clean_sheet_stats_df
        context = "clean sheet statistics"
    else:
        return None, None, None
    
    year_match = re.search(r'\b(20\d{2})\b', query)
    
    if year_match:
        year = year_match.group(1)
        df_filtered = df[df['Year'].str.contains(year)]
    else:
        df_filtered = df

    if df_filtered.empty:
        return None, None, None

    data_string = df_filtered.to_string(index=False)
    
    prediction_keywords = ['predict', 'prediction', 'forecast', 'likely to', 'expected to']
    is_prediction = any(keyword in query.lower() for keyword in prediction_keywords)
    
    return data_string, context, is_prediction

@app.route('/api/chat', methods=['POST'])
def chat():
    global previous_query, previous_response

    try:
        data = request.json
        user_input = data['message']
        
        # Log the received prompt
        print(f"Received prompt from frontend: {user_input}")
        
        previous_keywords = [
            'from the previous', 
            'from the last', 
            'from the prior', 
            'continue from the previous', 
            'based on the last', 
            'as mentioned before', 
            'as stated earlier', 
            'building on that', 
            'based on that',
            'following up on that', 
            'in addition to the previous', 
            'using the last', 
            'from earlier', 
            'from above', 
            'expand on the previous', 
            'extend the last response', 
            'carry on from before', 
            'follow up on the last', 
            'using what you just said', 
            'like you said before', 
            'elaborate on that', 
            'furthermore on that', 
            'add to the previous', 
            'continue where we left off', 
            'referring back to that', 
            'keeping that in mind', 
            'picking up from there', 
            'like mentioned before', 
            'referring to the earlier answer'
        ]
        
        is_previous = any(keyword in user_input.lower() for keyword in previous_keywords)

        # Check for a special request to use the previous response
        if is_previous:
            user_input = f"This was the previous response from you: {previous_response} and this was the previous user input: {previous_query}, now this is the current user input: {user_input}"
            print("Appending previous response to the current query.")

        # Check date relevance
        date_relevant = retrieve_with_pandas_date(user_input)
        
        # Get relevant data
        relevant_data, context, is_prediction = get_relevant_data(user_input)

        # Determine which model/approach to use
        if not date_relevant:
            completion = without_data(user_input)
        elif is_prediction:
            system_message = f"""You are a Premier League prediction expert. Using the following statistical data: {relevant_data}
            
            Provide a data-driven prediction for the query. Your response should:
            1. Clearly explain the basis of the prediction
            2. Use specific statistics from the provided data
            3. Quantify the prediction where possible
            4. Acknowledge any limitations or uncertainties in the prediction
            
            Prediction context: {context}
            
            User query: {user_input}"""
            
            completion = with_data(system_message, user_input)
        else:
            system_message = f"""You are a Premier League stat analyst. Use the following {context} data to answer the question: {relevant_data} 
            
            Provide a concise and accurate answer based solely on the data provided. 
            If the data doesn't contain the exact information needed, explain the limitations."""
            
            completion = with_data(system_message, user_input)

        ai_response = completion.choices[0].message['content']
        
        # Save to global variables
        previous_response = ai_response
        previous_query = user_input
        
        print(previous_response)
        print(previous_query)

        # Database logging remains the same
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)