from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    
)

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
    SystemData = "You need to optimize the following prompt, making it specifically easy for a generative ai to understand and perform better. Your response should only contain the rewritten prompt: "

    chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": SystemData},
        {"role": "user", "content": data}
    ],
    model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content



# start application with: uvicorn App:app --reload