#importing required libraries

from flask import Flask, request, render_template, make_response, jsonify
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from convert import convertion
import time
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

file = open("newmodel.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Add cache-busting headers to all responses
@app.after_request
def after_request(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
    return response

@app.route("/")
def home():
    timestamp = int(time.time())
    response = make_response(render_template("modern.html", timestamp=timestamp))
    return response

@app.route('/result',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        try:
            url = request.form["name"]
            print(f"Processing URL: {url}")
            
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1,30)
            
            y_pred = gbc.predict(x)[0]
            print(f"Prediction: {y_pred}")
            
            name = convertion(url, int(y_pred))
            print(f"Converted result: {name}")
            
            # Return JSON response for AJAX requests
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded':
                # Parse the result and return JSON
                result_data = {
                    'url': name[0],
                    'status': name[1],
                    'action': name[2],
                    'safe': len(name) > 3 and name[3] == "1"
                }
                return jsonify(result_data)
            
            # Fallback for regular form submissions
            timestamp = int(time.time())
            response = make_response(render_template("modern.html", name=name, timestamp=timestamp))
            return response
            
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            
            # Return JSON error for AJAX requests
            if request.headers.get('Content-Type') == 'application/x-www-form-urlencoded':
                error_data = {
                    'url': url if 'url' in locals() else "Unknown URL",
                    'status': "Error",
                    'action': "Unable to process",
                    'safe': False
                }
                return jsonify(error_data)
            
            # Fallback for regular form submissions
            error_result = [url if 'url' in locals() else "Unknown URL", "Error", "Unable to process", False]
            timestamp = int(time.time())
            response = make_response(render_template("modern.html", name=error_result, timestamp=timestamp))
            return response
    
    timestamp = int(time.time())
    return make_response(render_template("modern.html", timestamp=timestamp))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
