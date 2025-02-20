import numpy as np
from ultralytics import YOLO
import cv2
from src.model.load_model import model 
import traceback
from src.utils.dicom_utils import dicom_to_array

# Load YOLO model
def mask_output(pixel_array,model):

# Input and output image paths
# input_image_path = r"/home/omen/Documents/Megha/fracatlas/FracAtlas/test/images/IMG0003467.jpg"
# output_image_path = r"/home/omen/Documents/Megha/fracatlas/FracAtlas/infrencedimages/test1.jpg"
# mask_output_path = r"/home/omen/Documents/Megha/fracatlas/FracAtlas/infrencedimages/mask1.jpg"
    try:
# Load the image
        if len(pixel_array.shape) == 2:
            image = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
        else:
            image = pixel_array.copy()
        # image = cv2.imread(input_image_path)
        height, width, _ = image.shape
        mask = np.zeros((height,width), dtype=np.uint8)
        # Perform detection
        results = model.predict(image)
        print(results[0].boxes)

        # Annotate detections
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Bounding box coordinates
            # label = f"{model.names[int(cls)]} {conf:.2f}"  # Label with confidence
            mask[y1:y2, x1:x2] = 255 
            # # Draw bounding box and label
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)
            
            # cv2.imwrite(mask_output_path, mask)
#first convert the input image into a pixel array - that pixel array should be sent to model for the predictions= or can aslo save it to the jpg images and then xend to the model for prediction - 512 , 512v .
        return mask
    except Exception as e:
        # Enhanced error logging
        error_msg = f"""
        ERROR DETAILS:
        - Type: {type(e)}
        - Message: {str(e)}
        - Traceback: {traceback.format_exc()}
        """
        print(error_msg)  # This will appear in your FastAPI server logs
        return None
# Save the output image
