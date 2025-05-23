import os
import joblib

import litserve as ls
import pandas as pd




class InferenceAPI(ls.LitAPI):
    
    def setup(self, device="cpu"):
        self._model  = joblib.load("models/random_forest_model.pkl")
 
         

    def decode_request(self, request):
        try:
            columns = request ["columns"]
            rows = request ["data"]

            df = pd.DataFrame(rows, columns=columns)
            return df
        except Exception:
            return None

    def predict(self, x):
        print(x)
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        print(output, 9 * "*")
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response