from typing import Optional
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    print("asdasd")
    return {"Hello": "World"}

@app.get("/gps")
def read_item():
    return "asdsss"