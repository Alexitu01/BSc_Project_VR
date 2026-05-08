import runpod
from PIL import Image
from google.auth.transport.requests import Request
from datetime import datetime
import torch
import os
from io import BytesIO
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import FaceExtractor


# Download model weights on first run - writes to persistent volume
# Skipped if weights already exist from a previous run
def ensure_cacheDirectories():
    CacheList = ["/runpod-volume/huggingface/data_Cache", "/runpod-volume/huggingface/model_Cache", "/runpod-volume/tmp",]
    for directory in CacheList:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

async def handler(event):
    ensure_cacheDirectories()
    
    _input = event['input']
    #Turn the uploaded bytestring of the image into the image type that SPAG4D expects: 'Image' object from PIL
    _image = Image.open(BytesIO(base64.b64decode(_input.get('image'))))
    _image.load()

    #Define a unique name for the .ply file with the current time.
    filename = str(datetime.now().timestamp()) + ".png"
    
    #Save the image to the filename
    _image.save(filename)
    
    status = FaceExtractor.createPly(filename)
    
    if status['error'] == 1:
        return {"message": f"an error happened {status['message']}", "error": 1}
    
    try:
        #upload_to_drive will take a filename, and upload whatever object or file matches that filename
        upload_result = upload_to_drive("stitched_output.ply") #upload_result will contain the download url for the file once it has been uploaded to google drive
        
        #Since we use google drive, we dont need the file locally so we delete it.
        if os.path.exists("stitched_output.ply"):
            os.remove("stitched_output.ply")
            
        if os.path.exists(filename):
            os.remove(filename)
        
        return upload_result
    except Exception as e:
        if os.path.exists("stitched_output.ply"):
            os.remove("stitched_output.ply")
            
        if os.path.exists(filename):
            os.remove(filename)
        
        return {"message": str(e), "error": 1}

def get_drive_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/drive"],
        )
    
    creds.refresh(Request())

    #This constructs the resource that allows interaction with the google api.
    return build("drive", "v3", credentials=creds, cache_discovery=False,)


def upload_to_drive(filename):
    #Gets the id for the correct folder
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    
    try:
        service = get_drive_service()
        
        #Json format of the file's name, and what folder to put it in.
        #(Folder is put in an array, because 'parents' infers multiple folders)
        newFileName = str(datetime.now().timestamp()) + ".ply"
        meta_info = {"name": newFileName, "parents": [folder_id]}
        
        #This tells whatever gets this object, to wrap the bytes of the file from the filename
        # in an upload stream object so the file is sent in chunks instead of one big and heavy chunk.
        media = MediaFileUpload(filename, mimetype="application/octet-stream", resumable =True)
        #Creates files with the relevant information, and then returns the field "id".
        create_files = service.files().create(body = meta_info, media_body =media, fields="id").execute()
        file_id = create_files["id"]
        
        #After the file is uploaded it needs to change permissions on the file, so anyone can read it.
        service.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}).execute()

        #Return the file_id and the download url.
        return {
            "file_id": file_id,
            "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
            "error": 0
        }
    except Exception as e:
        return {"message": str(e), "error": 1}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
