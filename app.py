from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('AM_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
  
    
    column_names = ['Market Unit', 'Client Group', 'RBE', 'Industry', 'Platform',
       'ClientClassDesc', 'PricingStructureDesc', 'AREIRAGIndicator',
       'AREPRAGIndicator', 'SLAMRAGIndicator', 'AREI', 'AREP', 'SLAM',
       'TCR_MMS', 'SI_SGPct', 'CON_SGPct', 'IO_SGPct', 'BPO_SGPct', 'SC_SGPct',
       'LaborCostPct', 'CTDCostPct', 'CTDODECostPct', 'OnShoreHrsPct',
       'OnShoreCostASOfTotalCostPct', 'DNPAsOfTotalCostPct']

    df = pd.DataFrame(columns = column_names)
    final_features = df.append([pd.DataFrame([['Gallia', 'Gallia PRD', 'Technology', 'Industrial', 'ORACLE','Foundation','EFFORT-BASED: Time & Materials based on a flat ADR',
                                    #'False',
                                     'Others','Others','Others',
                                     4.21, 10.0, 100.0,2055.99202,0.0,0.0,0.0,0.0,0.0,0.645108,0.656306,0.656306,0.0,0.014169,0.000309
                         ]
                        ], 
                        columns=df.columns)])
    
    
    
                        
    prediction = model.predict(final_features)
    probability=model.predict_proba(final_features)[:,1] * 100
    output = prediction
    
    return render_template('index.html', prediction_text='PJ prediction is {},with probabiity of {}%'.format(output[0], probability[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    content = request.json
    column_names = ['Market Unit', 'Client Group', 'RBE', 'Industry', 'Platform',
       'ClientClassDesc', 'PricingStructureDesc', 'AREIRAGIndicator',
       'AREPRAGIndicator', 'SLAMRAGIndicator', 'AREI', 'AREP', 'SLAM',
       'TCR_MMS', 'SI_SGPct', 'CON_SGPct', 'IO_SGPct', 'BPO_SGPct', 'SC_SGPct',
       'LaborCostPct', 'CTDCostPct', 'CTDODECostPct', 'OnShoreHrsPct',
       'OnShoreCostASOfTotalCostPct', 'DNPAsOfTotalCostPct']
    df = pd.DataFrame(columns = column_names)
    final_features = df.append([pd.DataFrame([['Gallia', 'Gallia PRD', 'Technology', 'Industrial', 'ORACLE','Foundation','EFFORT-BASED: Time & Materials based on a flat ADR',
                                    #'False',
                                     'Others','Others','Others',
                                     4.21, 10.0, 100.0,2055.99202,0.0,0.0,0.0,0.0,0.0,0.645108,0.656306,0.656306,0.0,0.014169,0.000309
                         ]
                        ], 
                        columns=df.columns)])
                        
    #data = request.get_json(force=True)
    
    prediction = model.predict(final_features)
    probability=model.predict_proba(final_features)[:,1] * 100
    output = prediction

    prediction_text='PJ prediction is {},with probabiity of {}%, input json is {}'.format(output[0], probability[0],content)
    #return jsonify(output[0])
    return prediction_text

if __name__ == "__main__":
    app.run(debug=True)