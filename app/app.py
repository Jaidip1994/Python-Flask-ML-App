from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html', href = 'static/base_img.svg')
    else:
        text =  request.form['text']
        random_string = uuid.uuid4().hex
        predict_pic = f'static/{random_string}.svg'
        make_pic('AgesAndHeights.pkl', load('model.joblib'), float_string_to_arr(text), predict_pic)
        return render_template('index.html', href = predict_pic)

def make_pic(training_data_file_name, model, new_ip_np_arr, output_file):
    data = pd.read_pickle(training_data_file_name)
    data = data[data.Age > 0]
    x_new = np.array(list(range(19))).reshape(-1, 1)
    preds = model.predict(x_new)
    fig = px.scatter(x = data.Age, y = data.Height, title= "Heights vs Age of the People", labels= { 'x' : 'Age (Years)', 'y' : 'Height (inches)' })
    fig.add_trace(go.Scatter(x = x_new.reshape(-1), y = preds, mode= 'lines', name='Model'))
    new_pred = model.predict(new_ip_np_arr)
    fig.add_trace(go.Scatter(x=new_ip_np_arr.reshape(-1), y = new_pred, name = 'New o/p', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
    fig.write_image(output_file, width = 800)
    fig.show()
    
def float_string_to_arr(float_str):
    def isfloat(s):
        try:
            float(s)
            return True
        except:
            return False
    return np.array([float(elem) for elem in float_str.split(',') if isfloat(elem)]).reshape(-1, 1)