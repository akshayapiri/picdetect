# PicDetect – AI Image Classifier

A beautiful, modern web application that uses AI to classify images of animals and objects. Upload any image and get instant results with the object's name, category, confidence score, description, and fun facts!

## Features

- 🖼️ **Image Upload**: Drag & drop or click to upload images
- 🤖 **AI Classification**: Powered by AI to identify animals and objects
- 📊 **Confidence Scores**: See how confident the AI is in its classification
- 🎯 **Detailed Results**: Get name, category, and description
- 💡 **Fun Facts**: Learn interesting facts about the identified object
- 🎨 **Beautiful UI**: Modern, responsive design with smooth animations
- 📱 **Mobile Friendly**: Works great on all devices

## Getting Started

### Option 1: Run with Local Server (Recommended)

**Easy Way (Recommended):**
```bash
./start.sh
```
This starts both servers automatically. Then open `http://localhost:8000` in your browser.

**Manual Way:**
**Step 1: Start the API Proxy Server**
```bash
# Install dependencies (first time only)
pip3 install flask flask-cors pillow transformers torch

# Start the API proxy server
python3 api_proxy.py
```
This will run on `http://localhost:8001` and handle image classification.

**Step 2: Start the Web Server**
In a new terminal:
```bash
# Using Python
python3 server.py

# OR using Node.js
node server.js
```

**Step 3: Open in Browser**
Open `http://localhost:8000` in your browser.

**Note:** You need BOTH servers running:
- API Proxy (port 8001) - handles classification
- Web Server (port 8000) - serves the web app

### Option 2: Direct File (Limited)

You can open `index.html` directly, but API calls may be blocked by CORS. If you see CORS errors, use Option 1 instead.

### Using the App

1. Upload an image by clicking the upload area or dragging and dropping
2. Click "Classify Image" to analyze
3. View the results with all the details!

## Model Configuration

The app now runs image classification locally using Hugging Face transformers:

- Primary: `openai/clip-vit-base-patch32` in zero-shot mode with a rich set of labels (better at matching real-world object names such as *book*).
- Secondary fallback: `google/vit-base-patch16-224` and a color-based heuristic if the model download fails or the device is offline.

Weights download automatically on the first run and subsequent classifications happen entirely on your machine. To tweak accuracy, edit the candidate label list or model choice in `api_proxy.py`.

## File Structure

```
picdetect/
├── index.html      # Main HTML structure
├── style.css       # Styling and animations
├── script.js       # JavaScript logic and API integration
└── README.md       # This file
```

## Browser Compatibility

Works in all modern browsers that support:
- ES6 JavaScript
- Fetch API
- FileReader API
- CSS Grid and Flexbox

## Notes

- The app includes a fallback mock classification system if the API is unavailable
- Fun facts are stored locally and matched based on the classification results
- The app automatically selects appropriate emojis based on the detected category

Enjoy classifying images! 🎉

