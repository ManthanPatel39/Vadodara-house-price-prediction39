
from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np, os, joblib, traceback

app = Flask(__name__)
# Load dataset once to get all unique locations
DATA_PATH = "vadodara_house_data.csv"  # change to your dataset path

if os.path.exists(DATA_PATH):
    try:
        df_data = pd.read_csv(DATA_PATH)
        all_locations = sorted(df_data["location"].dropna().unique())
        print("Loaded locations:", len(all_locations))
    except Exception as e:
        print("Failed to load dataset:", e)
        all_locations = []
else:
    all_locations = []


MODEL_PATH = "model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Failed loading model:", e)

def format_lakh(x):
    try:
        v = float(x)
    except:
        return str(x)
    lakhs = v / 100000.0
    # show one decimal if <10, else one dec too
    if lakhs >= 1:
        return f"₹ {lakhs:,.1f} Lakh"
    else:
        # less than 1 lakh, show rupees
        return "₹ " + format(int(round(v)), ",d")

def parse_positive_number(val):
    if val is None:
        raise ValueError("Value missing")
    s = str(val).strip()
    if s in ("-","–","—"):
        raise ValueError("Invalid input '-'")
    s2 = s.replace(",","")
    try:
        num = float(s2)
    except:
        raise ValueError(f"Not a number: {s}")
    if num <= 0:
        raise ValueError("Value must be positive and greater than zero")
    return num

@app.route("/")
def home():
    return render_template("index.html")
@app.route('/charts')
def charts():
    return "<h2 style='text-align:center;color:#00c4b4;'>Charts Coming Soon...</h2>"
import csv
from datetime import datetime

@app.route("/contact", methods=["GET", "POST"])
def contact():
    msg = ""
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # Save message to a CSV file
        with open("messages.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, message])
        
        msg = f"✅ Thank you {name}, your message has been received successfully!"
    
    return render_template("contact.html", msg=msg)





@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", prediction_text="",locations=all_locations)
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        required = ['house_type','location','bhk','bathrooms','balcony','area_sqft']
        for f in required:
            if f not in data or str(data.get(f)).strip() == "":
                raise ValueError(f"Missing field: {f}")
        house_type = str(data.get('house_type')).strip()
        location = str(data.get('location')).strip()
        bhk = parse_positive_number(data.get('bhk'))
        bathrooms = parse_positive_number(data.get('bathrooms'))
        balcony = parse_positive_number(data.get('balcony'))
        area_sqft = parse_positive_number(data.get('area_sqft'))
        X = pd.DataFrame([{
            'house_type': house_type,
            'location': location,
            'bhk': bhk,
            'bathrooms': bathrooms,
            'balcony': balcony,
            'area_sqft': area_sqft
        }])
        if model is None:
            est = (area_sqft * 2500) + (bhk * 200000) + (bathrooms * 50000) + (balcony * 20000)
            pred = float(est)
            pred_text = format_lakh(pred)
        else:
            pred_val = model.predict(X)
            pred = float(pred_val[0])
            # if model predicts log values (very small), convert safely
            if pred < 50:
                try:
                    pred = float(np.expm1(pred))
                except:
                    pred = float(pred)
            pred_text = format_lakh(pred)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'pred_rupees': pred, 'pred_formatted': pred_text})
        return render_template("predict.html", prediction_text=pred_text)
    except Exception as e:
        err = str(e)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'error': err}), 400
        return render_template("predict.html", prediction_text="", error_message=err), 400

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
