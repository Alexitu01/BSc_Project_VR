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

from groq import Groq

googleClient = genai.Client()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
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
                    fixBottomAndTopCubes(faces)
                    print("*****************************************\n")
                    print("THIS IS THE PATH", imagePath)
                    print("*****************************************")
                    return {"response": imagePath, "error": 0}        
    return {"response": "There was an issue with the generated content, try again later", "error": 1}




def fixBottomAndTopCubes(path):
    faces = Convert.makeCubeMap(path)
    requiredFaces = ["top", "bottom", "left", "right", "back", "front"]

    for face in requiredFaces:
        if faces.get(face) is None:
            return {"response": f"Missing cubemap face: {face}", "error": 1}

    top_prompt = """Edit the first image only. The first image is the DOWN cubemap face.
The other four images are only reference images for continuity with the same 360° VR scene. Do not copy objects from them. Do not rearrange the scene. Do not invent a new floor, ground, room, road, or environment.
Return only one square image: the corrected DOWN cubemap face.
CRITICAL EDITING RULE:
Preserve the outer 25% border of the first image exactly as much as possible. The border area must stay visually the same, because it must connect to the neighboring cubemap faces. Do not move, remove, replace, or redesign objects, edges, walls, floor edges, terrain, shadows, furniture, lighting, or geometry near the edges.
Only improve the central 50% region of the DOWN face. Focus only on fixing broken, blurry, pinched, stretched, smeared, tripod-like, circular, or illogical geometry near the center of the downward-looking view.
The corrected center must blend naturally into the unchanged border. The result should look like the viewer is looking straight downward from the same fixed point in the same scene.
Maintain:
- same environment
- same objects
- same materials
- same lighting
- same perspective
- same scale
- same color tone
- same edge content
- same cubemap face orientation
Do not rewrite the whole image. Do not change the scene composition. Do not remove important objects visible near the edges. Do not change the left, right, top, or bottom borders of the first image.
Avoid:
new floor, new ground, new room, moved objects, missing objects, different lighting, different architecture, different materials, hard seams, circular hole, tripod artifact, pole pinching, swirl, smear, abstract texture, text, watermark, black borders.
Output only the repaired square DOWN cubemap face."""
    test = faces.get("up")
    
    
    response = googleClient.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[
        top_prompt,
        Image.open(faces["top"]),
        Image.open(faces["front"]),
        Image.open(faces["back"]),
        Image.open(faces["left"]),
        Image.open(faces["right"]),
    ])
    
    if response.parts != None:
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                image.save("fixedTop.png")
    else:
        return {"response": "There was an issue with the generated content, try again later", "error": 1}

    buttom_prompt = """Edit the first image only. The first image is the DOWN cubemap face.
The other four images are references only for continuity with the same 360° VR scene. Do not copy objects from them, do not rearrange the scene, and do not invent new objects, furniture, terrain, flooring, rooms, roads, props, or architecture.
Return only one square image: the corrected DOWN cubemap face.
Critical rule: preserve the outer 25% border of the first image as unchanged as possible. The border must remain visually the same because it connects to the neighboring cubemap faces. Do not move, remove, replace, redesign, or repaint objects, shadows, floor edges, walls, terrain, furniture, lighting, or geometry near the edges.
Only edit the central 50% region of the DOWN face. Focus only on repairing broken, blurry, pinched, stretched, smeared, circular, tripod-like, or illogical geometry near the center of the downward-looking view.
The repaired center must blend naturally into the unchanged border. The result must look like the viewer is looking straight down from the exact same fixed point in the same scene.
Preserve the same environment, objects, materials, lighting, perspective, scale, color tone, edge content, and cubemap orientation.
Do not rewrite the whole image. Do not change the scene composition. Do not remove important visible objects. Do not add new objects. Do not change the left, right, top, or bottom borders of the first image.
Avoid: new floor, new ground, new room, new road, new props, moved objects, missing objects, different lighting, different materials, different architecture, hard seams, circular hole, tripod artifact, pole pinching, swirl, smear, abstract texture, text, watermark, black borders.
Output only the repaired square DOWN cubemap face."""
    
    response = googleClient.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[
        buttom_prompt,
        Image.open(faces["bottom"]),
        Image.open(faces["front"]),
        Image.open(faces["back"]),
        Image.open(faces["left"]),
        Image.open(faces["right"]),
    ],)
    if response.parts != None:
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                image.save("fixedBottom.png")
    else:
        return {"response": "There was an issue with the generated content, try again later", "error": 1}

    
    
    return "It worked"



# start application with: uvicorn App:app --reload