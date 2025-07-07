from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load data
data = pd.read_csv("recipe_final.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

def recommend_recipes(input_features):
    # Scale the numerical input features
    input_features_scaled = scaler.transform([input_features[:7]])
    
    # Transform the ingredients input
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    
    # Combine the scaled numerical features with the transformed ingredients
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(input_combined, X_combined)
    
    # Get the indices of the top 5 most similar recipes
    indices = np.argsort(similarities[0])[::-1][:5]
    recommendations = data.iloc[indices]
    
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']]

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/recipe_form', methods=['GET', 'POST'])
def recipe_form():
    if request.method == 'POST':
        # Process form data and get recommendations
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']

        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)

        # Store recommendations in session
        session['recommendations'] = recommendations.to_dict(orient='records')
        return redirect(url_for('recommendations_page'))

    return render_template('index.html')

@app.route('/recommendations')
def recommendations_page():
    recommendations = session.get('recommendations', [])
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
