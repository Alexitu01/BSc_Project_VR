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

#Loads main page.
@app.get('/')
def index():
    path = "./FrontEnd/Pages/FrontPage.html"
    return FileResponse(path)

#Loads the 'creationpage' --> Name might be changed later, but currently as simple as possible.
@app.get('/CreationPage')
def creation():
    path = "./FrontEnd/Pages/CreationPage.html"


    return FileResponse(path)

#Post response from CreationPage, which takes the prompt written on the page, and sends it to the ai with instructions.
@app.post("/CreationPage")
async def read_item_via_request_body(request: Request):
    #The prompt received from the creationpage (see specification in the javascript)
    postRequest = await request.json()
    promptForAi = postRequest["prompt"]
    print(promptForAi)

    #System prompt containing instructions for the ai
    SystemData = "You need to optimize the following prompt, making it descriptive in a way that is easy for an image generation ai to understand and perform better. Your response should only contain the rewritten prompt: "

    #Call to the API with instructions and the prompt from the user.
    chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": SystemData},
        {"role": "user", "content": promptForAi}
    ],
    model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content



# start application with: uvicorn App:app --reload