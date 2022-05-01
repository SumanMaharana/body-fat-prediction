from flask import Flask,request,render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('reg.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    abdomen = float(request.form["abdomen"])
    chest = float(request.form["chest"])
    hip = float(request.form["hip"])
    weight = float(request.form["weight"])
    thigh = float(request.form["thigh"])
    prediction = model.predict([[abdomen,chest,hip,weight,thigh]])
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Your predicted body fat percentage:%{output}')


if __name__ == "__main__":
    app.run()