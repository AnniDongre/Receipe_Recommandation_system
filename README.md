# 🍽️ Recipe Recommendation System (Flask App)

A web-based **Recipe Recommendation System** implemented using **Flask** and **Unsupervised Learning** methods. The system suggests recipes from user-input nutritional values and ingredients. It employs **TF-IDF vectorization** and **cosine similarity** to provide similar recipes from a preprocessed dataset.

---

## 🔍 Key Features
- Web interface powered by **Flask**
- Provides recipe suggestions based on:
  - Nutrition (calories, fat, carbs, etc.)
  - Ingredients
- Uses **TF-IDF** for text features and **StandardScaler** for numerical features
- Recipe similarity is computed using **Cosine Similarity**
- Displays **Top 5** most similar recipes with:
  - Recipe name
  - Ingredients
  - Image

---

## 📊 Dataset Used

**FoodRecSys Dataset v1**  
📎 [Download from Kaggle](https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1)

> This dataset contains detailed recipe information such as ingredients, nutritional values, and image URLs.  
> ⚠️ Due to its size, it is **not included** in this repository. Please download it directly from Kaggle.

---

## 🛠️ Tech Stack
- Python
- Flask
- Pandas, NumPy
- Scikit-learn
- HTML + Jinja2 Templates

---

## 🔁 How It Works

```mermaid
flowchart TD
    A[User inputs nutrition values & ingredients] --> B[Vectorize ingredients using TF-IDF]
    B --> C[Scale numeric data using StandardScaler]
    C --> D[Combine features into a single vector]
    D --> E[Calculate Cosine Similarity with dataset]
    E --> F[Sort & select top 5 most similar recipes]
    F --> G[Display results with name, ingredients, and image]
