[!NOTE]
This project relies on several external APIs and service integrations, each of which requires locally configured credentials and environment-specific setup. There is no guarantee that the system will work unless all required keys, permissions, and dependencies have been configured appropriately.


# BSc Project VR

Proof-of-concept web application for generating panoramas from text prompts and converting a selected panorama into an explorable 3D Gaussian splat `.ply` file.

## Project Description

The project combines prompt optimization, image generation, and serverless 3D processing in a single workflow. A user can write a prompt, generate a panorama, select one image, and send it to a RunPod pipeline that converts the panorama into a stitched `.ply` file and returns a download URL.

## System Overview

The system consists of three main parts:

- `FrontEnd/`: browser-based user interface
- `App.py`: FastAPI application and orchestration layer
- `Projection/`: panorama-to-3D processing pipeline used by RunPod

At a high level, the system works like this:

1. The user enters a prompt in the frontend.
2. The prompt can be optimized through Groq.
3. A panorama image is generated through Gemini.
4. The selected panorama is sent to RunPod through the FastAPI backend.
5. RunPod processes the image into a stitched Gaussian splat `.ply` file.
6. The result is uploaded to Google Drive.
7. A download URL is returned to the frontend.

## Architecture

Main components:
- Frontend Web App
- FastAPI Backend
- Groq Prompt Optimization
- Gemini Image Generation
- RunPod Processing
- Google Drive Storage

## Installation

Install the root application dependencies:

```bash
pip install -r requirements.txt
```

The 3D pipeline in `Projection/` requires additional setup and separate environments for `ml-sharp` and `DA360`.

See [Projection/README.md](Projection/README.md) for the full pipeline setup.

## Configuration

The project depends on external services for prompt optimization, image generation, serverless processing, and file hosting.

Required services:
- Groq
- Google Gemini
- RunPod
- Google Drive

## Environment Variables

### Main application

- `GROQ_API_KEY`
- `SERVERLESS_API`

The Gemini client is created through `google.genai.Client()`, so valid Google credentials MUST be available in the environment.

## Running the System

Run the FastAPI app locally with Uvicorn:

```bash
uvicorn App:app --reload
```

## Limitations

- This project is a proof of concept
- The 3D reconstruction is based on a single panorama, so quality is limited
- Visual quality can degrade when moving far from the original viewpoint
- The 3D pipeline depends on external services, model weights, and multiple environments


## Authors / Credits

Authors:
Alexander Holst - alhh@itu.dk
Mathias Johnbeck - bamj@itu.dk
Símun Rasmussen - simra@itu.dk

OpenAI
Anthropic


This project builds on and integrates:
- `ml-sharp`
- `DA360`
- Groq
- Google Gemini
- RunPod
- Google Drive API



## References

Project-specific technical details for the panorama-to-3D pipeline are documented in `Projection/README.md`.
