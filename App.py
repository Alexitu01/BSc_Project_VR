from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.mount("/FrontEnd", StaticFiles(directory="FrontEnd"), name="FrontEnd")

@app.get('/')
def index():
    path = "./FrontEnd/Pages/FrontPage.html"
    return FileResponse(path)


# uvicorn App:app --reload