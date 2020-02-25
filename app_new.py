import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from flask import Flask, request, jsonify, render_template, send_file
import pickle
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
DOWNLOAD_FOLDER = './results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

@app.route("/")
def index():
    return render_template("test.html")


@app.route('/service_api',methods=['POST'])
def make_predict():    
    print(request.files)
    file = request.files['file1']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    service = pd.read_csv(app.config['UPLOAD_FOLDER'] + "/" + filename)
    model = pickle.load(open('service_model_en.pkl', 'rb'))
    service.replace('?', np.nan,inplace=True)
    cat_new=service.select_dtypes(include=object)
    cat_new.drop(["opened_time","updated_at","created_at","problem_ID", "change_request","problem_ID", "change_request"],axis=1,inplace=True)
    cat_new.drop(["Created_by"],axis=1,inplace=True)
    from sklearn.preprocessing import LabelEncoder
    df_temp = cat_new.astype("str").apply(LabelEncoder().fit_transform)
    df_final = df_temp.where(~cat_new.isna(), cat_new)
    new = service.filter(['count_reassign','count_updated'], axis=1)
    df=pd.concat([new,df_final],axis=1)
    df['category_ID']= df['category_ID'].fillna(df['category_ID'].value_counts().index[0])
    df['ID_caller'] = df['ID_caller'].fillna(df['ID_caller'].value_counts().index[0])
    df['location'] = df['location'].fillna(df['location'].value_counts().index[0])
    nonull=df[pd.isnull(df['opened_by'])==False]
    null=df[pd.isnull(df['opened_by'])]
    rf= RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',max_depth=None, max_features='auto', max_leaf_nodes=None,max_samples=None, min_impurity_decrease=0.0,min_impurity_split=None, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify']
    rf.fit(nonull[independent_var],nonull['opened_by'])
    prediction=rf.predict(X=null[independent_var])
    null['opened_by']=prediction.astype(int)
    data=nonull.append(null)
    nonull_1=data[pd.isnull(data['Support_group'])==False]
    null_1=data[pd.isnull(data['Support_group'])]
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify','opened_by']
    rf.fit(nonull_1[independent_var],nonull_1['Support_group'])
    prediction_1=rf.predict(X=null_1[independent_var])
    null_1['Support_group']=prediction_1.astype(int)
    nonull_1=data[pd.isnull(data['Support_group'])==False]
    null_1=data[pd.isnull(data['Support_group'])]
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify','opened_by']
    rf.fit(nonull_1[independent_var],nonull_1['Support_group'])
    prediction_1=rf.predict(X=null_1[independent_var])
    null_1['Support_group']=prediction_1.astype(int)
    data_1=nonull_1.append(null_1)
    nonull_2=data_1[pd.isnull(data_1['user_symptom'])==False]
    null_2=data_1[pd.isnull(data_1['user_symptom'])]
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify','opened_by','Support_group']
    rf.fit(nonull_2[independent_var],nonull_2['user_symptom'])
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify','opened_by','Support_group']
    rf.fit(nonull_2[independent_var],nonull_2['user_symptom'])
    prediction_2=rf.predict(X=null_2[independent_var])
    null_2['user_symptom']=prediction_2.astype(int)
    data_2=nonull_2.append(null_2)
    independent_var=['ID', 'ID_status', 'ID_caller', 'updated_by','type_contact', 'location', 'category_ID', 'notify','opened_by','Support_group','user_symptom']
    nonull_3=data_2[pd.isnull(data_2['support_incharge'])==False]
    null_3=data_2[pd.isnull(data_2['support_incharge'])]
    rf.fit(nonull_3[independent_var],nonull_3['support_incharge'])
    prediction_3=rf.predict(X=null_3[independent_var])
    null_3['support_incharge']=prediction_3.astype(int)
    Imputated_data=nonull_3.append(null_3)
    Imputated_data.drop(['ID_status','type_contact','type_contact','notify'],axis=1,inplace=True)
    pred = model.predict(Imputated_data)
    final_pred_file = pd.DataFrame(pred)
    df=pd.concat([service.ID,final_pred_file],axis=1)
    df.to_csv(os.path.join(app.config['DOWNLOAD_FOLDER'], 'result.csv'), index=False)
    return render_template('simple.html',  tables=[df.to_html(classes='data', header="true")])    

@app.route('/download', methods=['GET'])
def download_file():
    return send_file('./results/result.csv',as_attachment=True, attachment_filename='result.csv',mimetype='application/x-csv')

if __name__ == '__main__':
  app.run(host = 'localhost', port = 5000, debug = True)
