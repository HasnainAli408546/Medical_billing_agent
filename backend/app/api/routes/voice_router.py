import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from app.services.voice import voice_service
from app.schemas.voice import TTSRequest

router = APIRouter()

@router.post("/process")
async def process_voice(audio_file: UploadFile = File(...)):
    """
    Accepts an audio file upload (wav, mp3, etc.), writes to a temp file, 
    and transcribes it to text.
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save the uploaded byte-stream to a temporary file
    temp_audio_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        # Run STT Service
        # Transcribe is CPU bound, normally in FastApi we might run it on a thread pool,
        # but for simplicity we execute directly.
        transcription = voice_service.transcribe(temp_audio_path)
        
        return JSONResponse(content={"text": transcription})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
        
    finally:
        # Proper production cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@router.post("/speak")
async def generate_voice(request: TTSRequest):
    """
    Converts passed text into an audio file and returns it as a playable file stream.
    """
    temp_audio_path = os.path.join(tempfile.gettempdir(), f"output_{hash(request.text)}.mp3")
    
    try:
        await voice_service.generate_speech(request.text, temp_audio_path, request.voice)
        # BackgroundTasks could handle file deletion after response is sent
        return FileResponse(
            path=temp_audio_path, 
            media_type="audio/mpeg", 
            filename="response.mp3"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
