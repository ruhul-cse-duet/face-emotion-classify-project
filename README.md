# Face Emotion Classification

A modern, AI-powered web application for classifying Face Emotion images using deep learning. This application uses a custom ResNet CNN to classify images into five categories: ['Fear', 'Surprise', 'Angry', 'Sad', 'Happy']

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

## Live project link below
https://face-emotion-classify.streamlit.app/

## 🚀 Features

- **AI-Powered Classification**: Custom ResNet CNN for persons faces emotion images
- **Five Classes**: ['Fear', 'Surprise', 'Angry', 'Sad', 'Happy']
- **Modern UI**: Beautiful, responsive interface with gradient backgrounds
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Real-time Prediction**: Fast inference with confidence scores
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices
- 
## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run App.py
```


## Container Build & Run

1. Build the image:
   ```bash
   docker build -t face-emotion-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 --name face-emotion face-emotion-app
   ```
3. Open http://localhost:8501 in your browser.

If you need live code reloads during development, mount the repo as a volume:

```bash
docker run --rm -p 8501:8501 -v ${PWD}:/app face-emotion-app
```

## 📁 Project Structure

```
Lung and Colon Cancer Streamlit/
├── App.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── README.md                    # Project documentation
├── assets/
│   └── style.css                # Modern CSS styling
├── src/
│   ├── custom_resnet.py         # Model loading and prediction
├── models/
│   └── resnet_Model.pth         # Legacy/custom CNN weights
├── test_images/                 # Sample test images    
│   
└── Codes/
    └── face-emotion-classification.ipynb # Prior training notebook
```

After pushing, enable GitHub Actions or the repository’s container registry if you need automated builds. Update `README.md` with the image name/tag once it exists on Docker Hub or GHCR.

## 🔮 Future Enhancements

- [ ] Support for more Bone fracture types
- [ ] Integration with medical imaging systems
- [ ] Batch processing capabilities
- [ ] Advanced visualization tools
- [ ] API endpoints for integration
- [ ] Mobile app development
- [ ] Multi-language support

---
## Developer
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/);  
Email: ruhul.cse.duet@gmail.com
