from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    overallqual = int(request.form["overallqual"])
    grlivarea = int(request.form["grlivarea"])
    garagecars = int(request.form["garagecars"])
    totalbsmtsf = int(request.form["totalbsmtsf"])
    fullbath = int(request.form["fullbath"])
    yearbuilt = int(request.form["yearbuilt"])

    features = pd.DataFrame([{
        "OverallQual": overallqual,
        "GrLivArea": grlivarea,
        "GarageCars": garagecars,
        "TotalBsmtSF": totalbsmtsf,
        "FullBath": fullbath,
        "YearBuilt": yearbuilt
    }])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)