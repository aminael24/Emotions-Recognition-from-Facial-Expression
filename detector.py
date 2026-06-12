import cv2
import numpy as np
import tensorflow as tf

# Order MUST match alphabetical folder order used by flow_from_directory:
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTION_COLORS = {
    'angry':    (0,   0,   255),
    'disgust':  (0,   128, 0  ),
    'fear':     (128, 0,   128),
    'happy':    (0,   255, 255),
    'neutral':  (200, 200, 200),
    'sad':      (255, 100, 0  ),
    'surprise': (0,   165, 255),
}


class EmotionDetector:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._inp = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Haar cascade not found at {cascade_path}")

    def _preprocess(self, face_gray):
        face = cv2.resize(face_gray, (48, 48))
        face = face.astype(np.float32) / 255.0
        return face.reshape(1, 48, 48, 1)

    def predict(self, face_gray):
        """Returns (emotion_str, confidence_float)."""
        self.interpreter.set_tensor(self._inp['index'], self._preprocess(face_gray))
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self._out['index'])[0]
        idx = int(np.argmax(scores))
        return EMOTIONS[idx], float(scores[idx])

    def detect_faces(self, gray_frame):
        """Returns list of (x, y, w, h) bounding boxes."""
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
