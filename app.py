from flask import Flask, render_template, request
from model_predict import make_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "C:\\Users\\bansaln\\PycharmProjects\\HeartDisease\\"

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route("/", methods=['POST', 'GET'])
def submitted():
    if request.method == 'POST':
        result = request.form
        print(result)

        predict,predict_prob = make_predictions(result)
        if (predict_prob>=60):
            out_str="Warning! You've some heart condition, seek medical help ASAP!" \
                    "It is advised that you should:" \
                    " Get the following routine testings done:" \
                    " ECG (Electrocardiogram) " \
                    " ECHO(Echo cardiogram) " \
                    " Stress ECG." \
                    " IMMEDIATELY switch to better dietary habits; a low cholesterol, low sodium and no unhealthy fat diet." \
                    " Minimize stress and anxiety from your life; exercise, meditate and take plenty of sleep." \
                    " Get rid of smoking, alcohol and any other drug habits (if any) " \
                    "Prioritize cardiovascular exercises; try to run/jog/cycle/swim for atleast 25-30 mins daily."
        elif (predict_prob<=38):
            out_str="Yay, your heart health seems just fine! " \
                    "However, here are few tips to nullify the risk in future as well:" \
                    " Do cardiovascular exercises i.e. jogging, running, cycling and like atleast 4-5 times per week." \
                    " Quit smoking and alcoholism (if any)" \
                    " Keep up with good dietary and nutrition habits." \
                    "Cheers to a healthy heart!"
        else :
            out_str="Attention! You're a prone individual; you may develop a heart condition in future." \
                    "It is advised that, you should:" \
                    " Consult a certified cardiologist/medical practitioner." \
                    " Switch to a clean diet:" \
                    " Reduce potion size." \
                    "Take a low sodium and low cholesterol diet." \
                    " Get rid of unhealthy fats." \
                    " Eat more fruits, veggies and whole grains. " \
                    "Quit smoking/alcohol (if any)." \
                    " Manage stress/anxiety." \
                    "Get routine blood pressure checkups. " \
                    "Exercise regularly." \
                    "Research about healthy heart habits."
        return render_template("result.html",patientName=result['patientName'], predict =predict,predict_prob=predict_prob, out_str=out_str)

if __name__ == '__main__':
    app.run(debug=True)
