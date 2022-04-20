import pickle
from flask import Flask, render_template,request

app = Flask(__name__, template_folder='templates')


# load the model

model = pickle.load(open('diabetic_80.pkl','rb'))

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/data', methods = ['post'])
def data():
    preg = request.form.get('preg') 
    plas = request.form.get('plas')
    pres = request.form.get('pres')
    skin = request.form.get('skin')
    test = request.form.get('test') 
    mass = request.form.get('mass')
    pedi = request.form.get('pedi')
    age = request.form.get('age')

    result = model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])[0]

    data = {'preg':preg, 'plas':plas, 'pres':pres, 'skin':skin, 'test':test, 'mass':mass, 'pedi':pedi, 'age':age, 'result':result}
    return render_template('result.html', data=data)

app.run()