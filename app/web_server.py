"""
FastAPI web server for video translation with cloud GPU lip sync.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from app.config import config
from app.utils.logger import get_logger
from app.pipeline.process_video import VideoProcessor
from app.pipeline.lipsync_replicate import ReplicateLipSync

logger = get_logger(__name__)

app = FastAPI(title="Video Translation Platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

manager = ConnectionManager()

# Storage for processing jobs
processing_jobs: Dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        async with aiofiles.open(html_file, "r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Translation Platform</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                .upload-area {
                    border: 2px dashed #667eea;
                    border-radius: 10px;
                    padding: 40px;
                    text-align: center;
                    margin: 20px 0;
                    background: #f8f9ff;
                }
                .btn {
                    background: #667eea;
                    color: white;
                    padding: 12px 30px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                .btn:hover {
                    background: #5568d3;
                }
                #progress-container {
                    display: none;
                    margin: 20px 0;
                }
                .progress-bar {
                    width: 100%;
                    height: 30px;
                    background: #e0e0e0;
                    border-radius: 15px;
                    overflow: hidden;
                }
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    transition: width 0.3s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                }
                .status-text {
                    margin: 10px 0;
                    color: #666;
                }
                #results-container {
                    display: none;
                    margin: 20px 0;
                }
                .video-container {
                    margin: 20px 0;
                }
                video {
                    width: 100%;
                    max-width: 800px;
                    border-radius: 10px;
                }
                .download-links {
                    margin: 10px 0;
                }
                .download-links a {
                    display: inline-block;
                    margin: 5px;
                    padding: 8px 16px;
                    background: #667eea;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                .download-links a:hover {
                    background: #5568d3;
                }
                .language-select {
                    margin: 20px 0;
                }
                select {
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 Video Translation Platform</h1>
                <p style="text-align: center; color: #666;">
                    Upload your video and translate it to any language with AI-powered dubbing and lip sync
                </p>

                <div class="language-select">
                    <label>Target Language:</label>
                    <select id="language-select">
                        <option value="ES">Spanish</option>
                        <option value="DE">German</option>
                        <option value="FR">French</option>
                        <option value="IT">Italian</option>
                        <option value="PT">Portuguese</option>
                        <option value="RU">Russian</option>
                    </select>
                </div>

                <div class="upload-area">
                    <input type="file" id="video-upload" accept="video/*" style="display: none;">
                    <button class="btn" onclick="document.getElementById('video-upload').click()">
                        📁 Select Video File
                    </button>
                    <p id="file-name" style="margin-top: 10px; color: #666;"></p>
                </div>

                <div style="text-align: center;">
                    <button class="btn" id="process-btn" onclick="processVideo()" disabled>
                        🚀 Start Processing
                    </button>
                </div>

                <div id="progress-container">
                    <h3>Processing...</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                    </div>
                    <p class="status-text" id="status-text">Initializing...</p>
                </div>

                <div id="results-container">
                    <h2>✅ Processing Complete!</h2>
                    <div class="video-container">
                        <h3>Dubbed Video:</h3>
                        <video id="result-video" controls></video>
                    </div>
                    <div class="download-links">
                        <h3>Download Files:</h3>
                        <div id="download-buttons"></div>
                    </div>
                </div>
            </div>

            <script>
                let ws = null;
                let currentJobId = null;
                let selectedFile = null;

                document.getElementById('video-upload').addEventListener('change', function(e) {
                    selectedFile = e.target.files[0];
                    if (selectedFile) {
                        document.getElementById('file-name').textContent = `Selected: ${selectedFile.name}`;
                        document.getElementById('process-btn').disabled = false;
                    }
                });

                function connectWebSocket(jobId) {
                    ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);

                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateProgress(data);
                    };

                    ws.onerror = function(error) {
                        console.error('WebSocket error:', error);
                    };

                    ws.onclose = function() {
                        console.log('WebSocket closed');
                    };
                }

                function updateProgress(data) {
                    const progressFill = document.getElementById('progress-fill');
                    const statusText = document.getElementById('status-text');

                    if (data.progress !== undefined) {
                        progressFill.style.width = data.progress + '%';
                        progressFill.textContent = Math.round(data.progress) + '%';
                    }

                    if (data.status) {
                        statusText.textContent = data.status;
                    }

                    if (data.stage) {
                        statusText.textContent = `${data.stage}: ${data.status || ''}`;
                    }

                    if (data.completed) {
                        showResults(data);
                    }

                    if (data.error) {
                        alert('Error: ' + data.error);
                        document.getElementById('progress-container').style.display = 'none';
                    }
                }

                function showResults(data) {
                    document.getElementById('progress-container').style.display = 'none';
                    document.getElementById('results-container').style.display = 'block';

                    if (data.video_url) {
                        const video = document.getElementById('result-video');
                        video.src = data.video_url;
                    }

                    const downloadButtons = document.getElementById('download-buttons');
                    downloadButtons.innerHTML = '';

                    if (data.files) {
                        for (const [name, url] of Object.entries(data.files)) {
                            const link = document.createElement('a');
                            link.href = url;
                            link.textContent = `Download ${name}`;
                            link.download = name;
                            downloadButtons.appendChild(link);
                        }
                    }
                }

                async function processVideo() {
                    if (!selectedFile) {
                        alert('Please select a video file');
                        return;
                    }

                    const language = document.getElementById('language-select').value;
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('language', language);

                    document.getElementById('progress-container').style.display = 'block';
                    document.getElementById('results-container').style.display = 'none';
                    document.getElementById('process-btn').disabled = true;

                    try {
                        const response = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();
                        currentJobId = data.job_id;

                        connectWebSocket(currentJobId);
                    } catch (error) {
                        alert('Upload failed: ' + error.message);
                        document.getElementById('process-btn').disabled = false;
                    }
                }
            </script>
        </body>
        </html>
        """)


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    language: str = Form("ES")
):
    """Upload video and start processing."""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save uploaded file
        upload_dir = config.INPUT_PATH
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / f"{job_id}_{file.filename}"

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Create job record
        processing_jobs[job_id] = {
            "status": "pending",
            "file_path": str(file_path),
            "language": language,
            "created_at": datetime.now().isoformat()
        }

        # Start processing in background
        asyncio.create_task(process_video_job(job_id, file_path, language))

        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_job(job_id: str, video_path: Path, language: str):
    """Background task to process video (single-threaded, no multiprocessing)."""
    try:
        # Initialize processor with cloud lip sync
        # Note: Uses single-threaded processing, no WorkerPool
        lipsync = ReplicateLipSync()
        processor = VideoProcessor(lipsync=lipsync)

        # Send progress updates
        await manager.send_message(job_id, {
            "progress": 0,
            "stage": "Starting",
            "status": "Initializing video processing..."
        })

        # Process video with progress callbacks
        await manager.send_message(job_id, {
            "progress": 10,
            "stage": "Extracting Audio",
            "status": "Extracting audio from video..."
        })

        # Run in thread pool to avoid blocking event loop
        # But still single-process, no multiprocessing!
        import concurrent.futures
        loop = asyncio.get_event_loop()

        def process_sync():
            return processor.process_video(
                video_path,
                target_languages=[language]
            )

        # Run in thread (not process!) to avoid blocking
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, process_sync)

        if result.success:
            # Get output files
            video_file = result.dubbed_videos.get(language)
            audio_file = result.dubbed_audios.get(language)
            srt_file = result.translated_srts.get(language)

            await manager.send_message(job_id, {
                "completed": True,
                "progress": 100,
                "status": "Processing complete!",
                "video_url": f"/download/{video_file.name}" if video_file else None,
                "files": {
                    "video": f"/download/{video_file.name}" if video_file else None,
                    "audio": f"/download/{audio_file.name}" if audio_file else None,
                    "subtitles": f"/download/{srt_file.name}" if srt_file else None
                }
            })

            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["result"] = result.to_dict()

        else:
            await manager.send_message(job_id, {
                "error": result.error,
                "status": "failed"
            })
            processing_jobs[job_id]["status"] = "failed"

    except Exception as e:
        logger.error(f"Processing error for job {job_id}: {e}")
        await manager.send_message(job_id, {
            "error": str(e),
            "status": "failed"
        })
        processing_jobs[job_id]["status"] = "failed"


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed files."""
    # Check in output directory
    file_path = config.OUTPUT_PATH / filename
    if not file_path.exists():
        # Try finding in subdirectories
        for subdir in config.OUTPUT_PATH.iterdir():
            if subdir.is_dir():
                potential_path = subdir / filename
                if potential_path.exists():
                    file_path = potential_path
                    break

    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id in processing_jobs:
        return processing_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail="Job not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
