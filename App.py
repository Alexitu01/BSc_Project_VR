from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/FrontEnd", StaticFiles(directory="FrontEnd"), name="FrontEnd")

@app.get('/')
def index():
    path = "./FrontEnd/Pages/FrontPage.html"
    return FileResponse(path)

@app.get('/CreationPage')
def creation():
    path = "./FrontEnd/Pages/CreationPage.html"


    return FileResponse(path)

@app.post("/CreationPage")
async def read_item_via_request_body(request: Request):
    promptForAi = await request.json()
    data = promptForAi["text"]
    print(data)
    return data




# start application with: uvicorn App:app --reload