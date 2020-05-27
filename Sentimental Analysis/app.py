from flask import Flask, render_template,request
from tweets import AccessTwitter
import pandas as pd
import json
import os

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        search = 'covid-19'
        (z,x,y,c,d)=AccessTwitter(search)
        return render_template('index.html',emotion_plot = z,word_freqs=x,max_freq=y,sentiment_pie=json.dumps(c),table=json.dumps(d),search=search)
    else:
        search = request.form["srch-term"]
        (z,x,y,c,d)=AccessTwitter(search)
        return render_template('index.html',emotion_plot =z,word_freqs=x,max_freq=y,sentiment_pie=json.dumps(c),table=json.dumps(d),search=search)
   
   
   
port = int(os.environ.get('PORT', 5000))
app.run(host="0.0.0.0", port=port, debug=True)
#if __name__ == "__main__":
#    app.run(debug=True)