import os
import sys

import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file,obj):
    try:
        
        os.makedirs(os.path.dirname(file),exists_ok=True)
        
        with open(file,'wb') as f:
              pickle.dump(obj,f)
    except Exception as e:
             raise CustomException(e, sys)


def load_obj(file):
    try:
          with open("file",'rb') as f:
               pickle.load(f)
    except Exception as e:
             raise CustomException(e, sys)
    
def evaluate_model(Xtr,ytr,Xte,yte,models,params):
    try:
        model_report={}
        for i in range(len(models)):
                  model=list(models.values())[i]
                  model_name=list(models.keys())[i]
                  param=params[model_name]
                  gs=GridSearchCV(model,param,cv=3,scoring='accuracy', n_jobs=-1, verbose=1)
                  gs.fit(Xtr,ytr)

                  model.set_params(**gs.best_params_)
                  model.fit(Xtr,ytr)
                  ypred=model.predict(Xte)
                  acc_score=accuracy_score(yte,ypred)
                  model_report[model_name]=acc_score

                  print(f"[{model_name}] | Test: {acc_score:.4f}")
                  print(f"Best Params: {gs.best_params_}\n")

                  return model_report
        
    except Exception as e:
         CustomException(e,sys)


