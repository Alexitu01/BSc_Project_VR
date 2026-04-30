from datetime import datetime
import runpod
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
from pathlib import Path
from io import BytesIO
from google import genai
from google.genai.types import GenerateContentConfig, ImageConfig, Modality
from PIL import Image
from groq import Groq
import os
import time
import base64


googleClient = genai.Client()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
serverlessAPI = os.environ.get("SERVERLESS_API")

with open('instruction.txt', 'r') as file:
    imageSpecs = file.read().replace('\n', '')

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
    print(postRequest)
    #Determines if json contains a optomized prompt or an image prompt, and calls the corresponding function.
    if("prompt" in postRequest):
        return generatePrompt(postRequest["prompt"])
    elif "imagePrompt" in postRequest:
        return generateImages(postRequest["imagePrompt"])
    elif "imagePath" in postRequest:
        return generatePly(postRequest["imagePath"])
    else:
        return {"response": "Post request not supported", "error": 1}
    


def generatePly(imagePath):
    print(imagePath)
    with open(imagePath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data = {"image": encoded_string}

    endpoint = runpod.Endpoint("aamisvz3itx91m", serverlessAPI)

    run_request = endpoint.run(data)

    while True:
        status = run_request.status()
        print(status)
        
        if status == "COMPLETED":
            break
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError("Job ended with status: " + status)
        time.sleep(30)
    
    return run_request.output() 


def generatePrompt(promptForAi):
    print(promptForAi)

    #System prompt containing instructions for the ai
    SystemData = """You need to optimize the following prompt, making it specifically able to convey more details and convey a feeling of a full 360 degree environment. 
    It must be able to describe an equirectangular projection in such a way that enhances the performance of a 3d image generation ai.
    (This means the generated environment from your prompt, should be able to show a full 360 view around the viewer, while also being able to convey a full 360 view from top to bottom)
    Your response should only contain the rewritten prompt: """

    #Call to the API with instructions and the prompt from the user.
    chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": SystemData},
        {"role": "user", "content": promptForAi}
    ],
    model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

def generateImages(promptForAi):
    try:
        print("*********Generation begin************")
        response = googleClient.models.generate_content( #Calls the googleClient, with specified prompt for generating image
            model="gemini-3.1-flash-image-preview",
            contents=(promptForAi),
            config=GenerateContentConfig(
                system_instruction=[imageSpecs], #Containst system instructions (see instruction.txt)
                response_modalities=[Modality.TEXT, Modality.IMAGE],
                image_config = ImageConfig(aspect_ratio="21:9")
            ),
        )
    except:
        return {"response": "An issue emerged from trying to generate the image \n The problem can stem from: \n 1. Quota has been met (wait 1m between every generation) \n 2. There was an issue with authorization ", "error": 1 }
        
    print("Finished generating")
    if response.candidates != None and response.candidates[0].content != None and response.candidates[0].content.parts != None: #Necessary checks or else compiler cries
        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text) #The 'thinking' text that the AI responds with in its process of generating an image.
            elif part.inline_data:
                if part.inline_data.data != None:
                    image = Image.open(BytesIO((part.inline_data.data))) #Defines the reference for the image generated
                    output_dir = "FrontEnd/Images"
                    imagePath = os.path.join(output_dir, str(datetime.now()) + ".png") #Using datetime to give a unique name to each image generated.
                    image.save(imagePath) #Saves the image to the filepath
                    print("*****************************************\n")
                    print("THIS IS THE PATH", imagePath)
                    print("*****************************************")
                    return {"response": imagePath, "error": 0}                
    return {"response": "There was an issue with the generated content, try again later", "error": 1}

# start application with: uvicorn App:app --reload