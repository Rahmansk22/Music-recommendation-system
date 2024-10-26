from keras.models import load_model
import numpy as np
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Path to the Haar cascade classifier file
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

# Path to the emotion detection model file
emotion_model_path = 'final_model.h5'

# Emotion labels
EMOTIONS = ["happy", "sad","energetic","cry"]

# Load the Haar cascade classifier for face detection
face_detection = cv2.CascadeClassifier(detection_model_path)

# Load the emotion detection model
try:
    emotion_classifier = load_model(emotion_model_path, compile=False)
except Exception as e:
    print("Error loading emotion detection model:", e)
    exit()

# Function for real-time facial emotion analysis
def emotion_testing():
    try:
        cap = cv2.VideoCapture(0)
        while True:
            ret, test_img = cap.read()
            if not ret:
                continue
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]  # Cropping region of interest (face area) from the image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = emotion_classifier.predict(img_pixels)

                # Find the predicted emotion
                max_index = np.argmax(predictions[0])
                predicted_emotion = EMOTIONS[max_index]

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the resized image with emotion analysis
            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis', resized_img)

            # Press 'q' to quit the loop and close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return predicted_emotion
    except Exception as e:
        print("An error occurred during emotion testing:", e)
        return None

# Call the emotion_testing function
emotion_word = emotion_testing()
if emotion_word is not None:
    print("Detected emotion:", emotion_word)
else:
    print("Error occurred during emotion testing.")
