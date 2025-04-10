import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import torchvision.models.detection as detection
import numpy as np

# Define the class names
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Load the trained model
def load_trained_model(model_path, num_classes):
    model = detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Function to predict on a single frame
def predict(image_tensor, model, device):
    model.to(device)
    image_tensor = [image_tensor.to(device)]
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

# Draw predictions on the frame
def draw_predictions(frame_tensor, predictions, threshold=0.9):
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    class_names = [classes[i] for i in labels]
    texts = [f'{name}: {score:.2f}' for name, score in zip(class_names, scores)]

    drawn = draw_bounding_boxes(frame_tensor, boxes, texts, colors='red', width=3)
    return drawn.permute(1, 2, 0).cpu().numpy()

# Load the model
model_path = 'fasterrcnn_fruit_detect_statedict.pth'  # Replace with your model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model(model_path, num_classes)

# Set up transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

print("Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB and then to PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Transform to tensor
    image_tensor = transform(pil_image)

    # Run prediction
    predictions = predict(image_tensor, model, device)

    # Draw predictions
    drawn = draw_predictions((image_tensor * 255).byte(), predictions)

    # Convert back to BGR for OpenCV display
    bgr_output = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)

    # Show the frame
    cv2.imshow('Live Fruit Detection', bgr_output)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
