from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from io import BytesIO
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
model_path = "C:\\Users\\preci\\Downloads\\maizediseasemodel_final.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Maize disease classes
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Load model
def load_model():
    """Load the pretrained maize disease model"""
    try:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Initialize model at startup
try:
    model = load_model()
except RuntimeError as e:
    print(f"Warning: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Maize Disease Detection API. Use /upload-photo/ to POST an image."}  
@app.post("/upload-photo/")
async def upload_photo(file: UploadFile = File(...)):
    """
    Upload, preprocess, and predict image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with prediction results and confidence scores
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are supported"
            )
        
        # Check if model is loaded
        if 'model' not in globals():
            raise HTTPException(
                status_code=500,
                detail="Model not loaded. Please check model path."
            )
        
        # Read file content
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Get original dimensions
        original_size = image.size
        # Apply preprocessing
        image_tensor = preprocess(image)
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Get predictions
        pred_idx = predicted_class.item()
        pred_class = class_names[pred_idx]
        pred_confidence = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        class_probabilities = {
            class_names[i]: float(all_probs[i])
            for i in range(len(class_names))
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Image processed and prediction completed",
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "processed_shape": list(image_batch.shape),
                "prediction": {
                    "class": pred_class,
                    "confidence": round(pred_confidence, 4),
                    "all_probabilities": {k: round(v, 4) for k, v in class_probabilities.items()}
                },
                "device": str(device)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
