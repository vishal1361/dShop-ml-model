from fastapi import FastAPI, HTTPException
from fastapi.params import Query
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import List
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD
from collections import defaultdict

app = FastAPI()

# Define request body model
class RecommendationRequest(BaseModel):
    userId: str
    count: int
    model: int
    weight: int

# Load all the data and functions
try:
    loaded_dataframe = pickle.load(open('dataframe.pkl', 'rb'))
    model_1 = pickle.load(open('Model1.sav', 'rb'))
    model_2 = pickle.load(open('Model2.sav', 'rb'))
    # get_recommendations = pickle.load(open('get_recommendations.pkl', 'rb'))
    # rep_rec_model = pickle.load(open('reputation_recommendation_model.pkl', 'rb'))

except FileNotFoundError:
    raise RuntimeError("One or more pickle files not found. Please check the file paths.")

@app.get('/recommendations', response_model=List[str], tags=['Recommendations'])
def recommend_using_get_call(userId: str = Query(..., description="User ID for which recommendations are requested"),
                             count: int = Query(10, description="Number of recommendations to return"),
                             model: int = Query(1, description="Model number to use for recommendations"),
                             weight: float = Query(0.5, description="Weight to apply to the recommendations")):
    """
    Get recommendations based on user reputation.
    """
    # Check if user_id is valid
    if not userId:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    # Placeholder for model validation
    if model not in [1, 2]:
        raise HTTPException(status_code=400, detail="Invalid model number")

    # Placeholder for weight validation
    if weight <= 0:
        raise HTTPException(status_code=400, detail="Weight must be a positive integer")

    # Select model based on user's choice
    selected_model = model_1 if model == 1 else model_2

    # Call the method to get recommendations
    recommendations = weighted_borda_count_df(selected_model, loaded_dataframe, userId, count, weight/10)
    # Convert the list of recommendations to a single string separated by commas
    recommendations_str = ",".join(recommendations)
    # Return the list containing the string of recommendations
    return [recommendations_str]

@app.post('/recommendations', response_model=List[str], summary="Get recommendations based on user reputation")
def recommend(recommendation_request: RecommendationRequest):
    """
    Get recommendations based on user reputation.

    - **userId**: User ID for which recommendations are requested.
    - **model**: Model number to use for recommendations.
    - **count**: Number of recommendations to return.
    - **weight**: Weight to apply to the recommendations.
    """
    # Extract user ID, model, count, and weight from request
    user_id = recommendation_request.userId
    count = recommendation_request.count
    model = recommendation_request.model
    weight = recommendation_request.weight

    # Check if user_id is valid (you may have your own validation logic)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    # Placeholder for model validation
    if model not in [1, 2]:
        raise HTTPException(status_code=400, detail="Invalid model number")

    # Placeholder for weight validation
    if weight <= 0:
        raise HTTPException(status_code=400, detail="Weight must be a positive integer")

    # Select model based on user's choice
    selected_model = model_1 if model == 1 else model_2

    # Call the method to get recommendations from rep_rec_model
    recommendations = weighted_borda_count_df(selected_model, loaded_dataframe, user_id, count, weight/10)
    return recommendations

def get_recommendations(data, user_id, top_n, algo):

    # Creating an empty list to store the recommended product ids
    recommendations = []

    # Creating an user item interactions matrix
    user_item_interactions_matrix = data.pivot(index = 'user_id', columns = 'prod_id', values = 'rating')

    # Extracting those product ids which the user_id has not interacted yet
    non_interacted_products = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    # Looping through each of the product ids which user_id has not interacted yet
    for item_id in non_interacted_products:

        # Predicting the ratings for those non interacted product ids by this user
        est = algo.predict(user_id, item_id).est

        # Appending the predicted ratings
        recommendations.append((item_id, est))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key = lambda x: x[1], reverse = True)

    return recommendations[:top_n] # Returing top n highest predicted rating products for this user

def weighted_borda_count_df(model, df,  userId, count, weight):
    recommendations = get_recommendations(df, userId, 100, model)
    temp_df = pd.DataFrame(recommendations, columns=['prod_id', 'predicted_ratings'])

    # Iterate over the rows of temp_df
    for index, row in temp_df.iterrows():
        # Extract the product ID from the current row of temp_df
        prod_id = row['prod_id']

        # Find the corresponding row in df_merged where prod_id matches
        matching_row = df[df['prod_id'] == prod_id]

        # Check if a matching row is found
        if not matching_row.empty:
            # Extract the seller_reputation value from df_merged and set it in temp_df
            temp_df.at[index, 'seller_reputation'] = matching_row['seller_reputation'].values[0]

    # Sort by predicted_ratings and assign ranks
    temp_df = temp_df.sort_values(by=['predicted_ratings', 'seller_reputation'], ascending = False)
    temp_df['rec_borda'] = temp_df['predicted_ratings'].argsort().argsort() + 1

    # Sort by seller_reputation and assign ranks
    temp_df = temp_df.sort_values(by=['seller_reputation', 'predicted_ratings'], ascending = False)
    temp_df['rep_borda'] = temp_df['seller_reputation'].argsort().argsort() + 1

    # Merge the BCs of the same item in the two lists by summing them up
    temp_df['merged_borda'] = temp_df['rec_borda'] + temp_df['rep_borda']

    # Weight the BCs from the recommendation list
    temp_df['weighted_rec_borda'] = weight * temp_df['rec_borda']

    # Weight the BCs from the reputation list
    temp_df['weighted_rep_borda'] = (1 - weight) * temp_df['rep_borda']

    # Combine the weighted BCs
    temp_df['weighted_merged_borda'] = temp_df['weighted_rec_borda'] + temp_df['weighted_rep_borda']

    # Sort the items based on their final scores
    temp_df = temp_df.sort_values(by='weighted_merged_borda', ascending=False)
    print(temp_df)
    return temp_df["prod_id"][:count]
