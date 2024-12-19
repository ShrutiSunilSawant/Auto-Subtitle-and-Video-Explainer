# How to Run Auto Subtitle and Video Explainer

## Setup Instructions

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/MacOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install FFmpeg
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# On Windows (using Chocolatey)
choco install ffmpeg

# On MacOS (using Homebrew)
brew install ffmpeg
```

### 3. Project Structure
Ensure your project has the following structure:
```
project_root/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Dependencies file
├── static/               # For processed videos
├── templates/            # HTML templates
│   └── upload.html       # Upload interface
└── README.md            # Project documentation
```

### 4. Running the Application

```bash
# Start the Flask server
python app.py
```

The application will start on `http://localhost:5006`

## Usage Instructions

1. Open your web browser and navigate to `http://localhost:5006`
2. Select your preferred language and subtitle size
3. Upload your video file
4. Wait for processing to complete
5. Download the processed video with subtitles

## Troubleshooting

### Common Issues and Solutions

1. **GPU Not Detected**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **FFmpeg Missing**
   ```bash
   # Verify FFmpeg installation
   ffmpeg -version
   ```

3. **Memory Issues**
   - Reduce video resolution
   - Close other applications
   - Increase system swap space

4. **Model Download Issues**
   - Check internet connection
   - Verify disk space
   - Try manual model download

### Error Messages

1. **"CUDA out of memory"**
   - Reduce batch size
   - Process shorter video segments

2. **"FFmpeg not found"**
   - Reinstall FFmpeg
   - Add to system PATH

3. **"Port already in use"**
   ```bash
   # Kill process using port 5006
   # On Linux/MacOS
   lsof -i:5006
   kill -9 <PID>

   # On Windows
   netstat -ano | findstr :5006
   taskkill /PID <PID> /F
   ```

## Development Notes

### Running in Debug Mode
```bash
# Enable debug mode
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Testing Video Processing
```bash
# Test with a small video first
# Recommended test video specs:
# - Duration: 30 seconds
# - Resolution: 720p
# - Format: MP4
```

### Performance Optimization
- Use GPU when available
- Process in batches
- Monitor memory usage

## Security Notes

1. Implement proper input validation
2. Set up file size limits
3. Configure CORS if needed
4. Use secure headers

## Backup and Recovery

1. Processed videos are stored in `static/`
2. Implement regular cleanup of old files
3. Monitor disk space usage

## Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)