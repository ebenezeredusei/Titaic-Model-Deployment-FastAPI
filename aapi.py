from fastapi import FastAPI
import uvicorn
import pickle
from joblib import load
from pydantic import BaseModel, Field
import pandas as pd
from os.path import exists
from sklearn.ensemble import RandomForestClassifier

"""#model = load('model.joblib', 'rb')
# load the saved model
loaded_model = pickle.load(open('trained_rfc_model.sav', 'rb'))

    
class PassengerModel(BaseModel):
    Sex: int
    Age: int 
    Fare: float 
    Pclass1: bool 
    Pclass2: bool 
    Pclass3: bool
    Embarked_S: bool 
    Is_alone_True: bool
    
    
app = FastAPI()

@app.get("/")
async def predict(data: PassengerModel):
    pred_data = pd.DataFrame([data.dict()])
    prediction = model.predict_prob(pred_data)
    return_val = prediction[0][0]
    #print(pred_data)
    return {'prediction': return_val}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)"""
    
    
'''class PassengerModel(BaseModel):
    Sex: int
    Age: int 
    Fare: float 
    Pclass1: int 
    Pclass2: int 
    Pclass3: int
    Embarked_S: int 
    Is_alone_True: int'''
    
app = FastAPI()
    
class PassengerModel(BaseModel):
    Sex: int
    Age: int 
    Fare: float 
    Pclass1: int 
    Pclass2: int 
    Pclass3: int
    Embarked_S: int 
    Is_alone_True: int


def home():
    return {'text': 'Titanic Survival Prediction'}


@app.post('/predict', response_model=PassengerModel)
def preprocess_data(data: PassengerModel):
    	
	
    # male/female indicators
	Sex=0
 
    # Age indicators
	Age=0
 
    # fare indicators
	Fare=0
 
    # passenger class
	Pclass_1=0
	Pclass_2=0
	Pclass_3=0
	
	    
	# embarked indicators
	Embarked_S=0
	
    # Is_Alone indicators
	Is_alone_True=0
	
	# store preprocessed data / model input as dictionary
	processed = {"Sex": Sex,
                "Age": Age,
                "Fare": Fare,
                "Pclass_1": Pclass_1,
                "Pclass_2": Pclass_2,
                "Pclass_3": Pclass_3,
                "Embarked_at_Southampton": Embarked_S,
                "Is_alone": Is_alone_True}
                
        # return preprocessed data
	return processed

# define post method to get prediction from model
@app.post("/predict/")
def predict_from_preprocessed(data: PassengerModel):

	# get data from request body
	inputData = data.dict()
	# convert to pd DF since sklearn cannot predict from dict
	inputDF = pd.DataFrame(inputData, index=[0])
	
	# load the Random Forest Classifier
	with open("./trained_rfc_model.sav", 'rb') as file:
            rf = pickle.load(file)
            
            SurvivalProba = rf.predict_proba(inputDF)[0,1]
            survPerc = round(SurvivalProba*100, 1)
            
            return survPerc
 
    
if __name__ == '__main__':
    uvicorn.run(app)
