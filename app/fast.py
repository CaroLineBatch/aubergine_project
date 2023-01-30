from fastapi import FastAPI , File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mlops.model import load , predict
from mlops.preprocess import preprocess


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

MODEL= load()

@app.get("/")
def root():
    return {'greeting': 'Hello'}


#http://0.0.0.0:8000/pred
@app.post("/pred")
async def create_upload_file(file: bytes = File()):
    pred = predict(MODEL , file)
    #return 0 if dick 1 if no dick
    return {'prediction' : f'{int(pred)}'}
