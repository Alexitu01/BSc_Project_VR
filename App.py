from datetime import datetime
from Projection import Convert
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
from io import BytesIO
from google import genai
from google.genai.types import GenerateContentConfig, ImageConfig, Modality
from PIL import Image
import os
import time

from groq import Groq

googleClient = genai.Client()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
with open('instruction.txt', 'r') as file:
    imageSpecs = file.read().replace('\n', '')

with open('fixTop.txt', 'r') as fixTop:
    fixTopInstructions = fixTop.read()

with open('fixDown.txt', 'r') as fixDown:
    fixDownInstructions = fixDown.read()

app = FastAPI()
app.mount("/FrontEnd", StaticFiles(directory="FrontEnd"), name="FrontEnd")
waits = [300, 300, 600, 600]


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
    
    #Determines if json contains a optomized prompt or an image prompt, and calls the corresponding function.
    if("prompt" in postRequest):
        return generatePrompt(postRequest["prompt"])
    else:
        return generateImages(postRequest["image"])




def generatePrompt(promptForAi):
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
                    faces = Convert.makeCubeMap(imagePath)
                    fixTopFace(faces)
                    print("*****************************************\n")
                    print("THIS IS THE PATH", imagePath)
                    print("*****************************************")
                    return {"response": imagePath, "error": 0}        
    return {"response": "There was an issue with the generated content, try again later", "error": 1}




def fixTopFace(faces):
    errorlist = []
    requiredFaces = ["top", "bottom", "left", "right", "back", "front"]

    for face in requiredFaces:
        if faces.get(face) is None:
            return {"response": f"Missing cubemap face: {face}", "error": 1}

    attempt = 0
    for attempt in range(3):
        try:
            response = googleClient.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                str(fixTopInstructions),
                Image.open(faces["top"]),
            ])
            print(f"Response received at repetition: {attempt}")
            if response.parts != None:
                for part in response.parts:
                    if part.text is not None:
                        print(part.text)
                    elif image:= part.as_image():
                        image.save("fixedTop.png")
                        print("Fixed Top was generated")
                        break
                break
            else:
                return {"response": "There was an issue with the generated content, try again later", "error": 1}
        except Exception as e:
            print(f"Attempt number: {attempt + 1} failed, sleeping and retrying.")
            time.sleep(waits[attempt])
            attempt = attempt + 1
            errorlist.append((e.args, str(datetime.now())))
            print(f"Starting repetition for UP")
    
    return "fixedTop.png"
            
    

    
def fixBottomFace(faces):
    errorlist = []
    attempt = 0
    for attempt in range(2):
        try:
            response = googleClient.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                fixDownInstructions,
                Image.open(faces["bottom"]),
            ],)
            print(f"Response received at repetition: {attempt}")
            if response.parts != None:
                for part in response.parts:
                    if part.text is not None:
                        print(part.text)
                    elif image:= part.as_image():
                        image.save("fixedBottom.png")
                        print("fixed Bottom was generated")
                        break
                break
            else:
                return {"response": "There was an issue with the generated content, try again later", "error": 1}
        except Exception as e:
            time.sleep(waits[attempt])
            attempt = attempt + 1
            errorlist.append((e.args, str(datetime.now())))
            print(f"Starting repetition for DOWN number {attempt}")
   
    for error, timestamp in errorlist:
        print(f"Error: {error}")
        print(f"Time: {timestamp}")

    return "fixedBottom.png"

# start application with: uvicorn App:app --reload