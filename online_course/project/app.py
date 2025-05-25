from flask import Flask, request, jsonify, render_template
from model import hybrid_recommend

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form.get("user_id"))
    recommendations = hybrid_recommend(user_id)
    return render_template("index.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
