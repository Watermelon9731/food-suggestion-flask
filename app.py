from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import data_loader

app = Flask(__name__)
CORS(app)

data = data_loader.init()

keyword_list = data['name'].dropna().tolist()

encoder_food_type = LabelEncoder()
encoder_food_flavour = LabelEncoder()
encoder_food_meal = LabelEncoder()
encoder_food_process = LabelEncoder()
encoder_food_vegetarian = LabelEncoder()
encoder_food_name = LabelEncoder()

data["type"] = encoder_food_type.fit_transform(data["type"])
data["flavour"] = encoder_food_flavour.fit_transform(data["flavour"])
data["meal"] = encoder_food_meal.fit_transform(data["meal"])
data["process"] = encoder_food_process.fit_transform(data["process"])
data["vegetarian"] = encoder_food_vegetarian.fit_transform(data["vegetarian"])
data["name"] = encoder_food_name.fit_transform(data["name"])

X = data[["type", "flavour", "meal", "process", "vegetarian"]]
Y = data[["name"]]
decision_model = DecisionTreeClassifier()
decision_model.fit(X, Y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        print(user_data)
        food_type = encoder_food_type.transform([user_data["type"]])[0]
        food_flavour = encoder_food_flavour.transform([user_data["flavour"]])[0]
        food_meal = encoder_food_meal.transform([user_data["meal"]])[0]
        food_process = encoder_food_process.transform([user_data["process"]])[0]
        food_vegetarian = encoder_food_vegetarian.transform([user_data["vegetarian"]])[0]
        print('inside', food_type, food_flavour, food_process)
        input_data = [[food_type, food_flavour, food_meal, food_process, food_vegetarian]]
        prediction = decision_model.predict(input_data)
        dish_name = encoder_food_name.inverse_transform(prediction)[0]

        return jsonify({"recommendation": dish_name})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/search', methods=['POST'])
def search():
    try:
        search_data = request.json
        all_keywords = [search_data['keyword']] + keyword_list

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_keywords)

        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        most_similar_idx = np.argmax(cos_sim)
        most_similar_score = cos_sim[0][most_similar_idx]
        most_similar_keyword = keyword_list[most_similar_idx]
        return jsonify({"most_similar_keyword": most_similar_keyword, "most_similar_score": most_similar_score})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
