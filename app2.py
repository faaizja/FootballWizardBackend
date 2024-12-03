# import openai
# import pandas as pd
# import os

# # Use RAGS here first, take the user input, embed the question and have documents with 2020, 2021, 2022, 2023, 2020/21, 2020/2021, 2021/22, 2021/2022, 2022/23, 2022/2023, 2022/23, 2023/2024, 2024/25, 2024/2025
# # we will check to see if the input has any of these years/seaons, if so we will then run another RAGS looking for goal/goals, assist/assists, clean sheets, etc and then we will determind which CSV file we need 
# # to give as context to the model before we send, then we will call either function model_without(userInput) or model_with(userInput, csv) and we will get a response from the model using either our data or the pre 
# # existing data from openai prior to 2021.

# # we could also have two models, one for stats and analysis and the other for predictions where we can make {"role": "system", "content": "You are a premier leauge stat analyst. You can also predict the furture if asked!"}, this like different

# df = pd.read_csv('goal_stats.csv')

# top_scorers_data = df.to_string(index=False)

# openai.api_key = os.getenv("OPENAI_API_KEY")

# user_input = input("Please enter your question: ")

# # {"role": "user", "content": f"Analyze these goal scorers and tell me who has the most goals in 2023:\n{top_scorers_data}"}

# completion = openai.ChatCompletion.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a premier leauge stat analyst. You can also predict the furture if asked!"},
#         {"role": "system", "content": f"Using this data:\n{top_scorers_data} either give the analysis or the prediction"},
#         {"role": "user", "content": user_input}
#     ]
# )

# print(completion.choices[0].message['content'])