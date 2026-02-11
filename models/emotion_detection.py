from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        pass  # No model to initialize in DeepFace, it loads on first use

    def detect(self, face_image):
        try:
            # Analyze emotion with DeepFace
            analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
            # Get the dominant emotion
            emotion = analysis[0]['dominant_emotion']
            return emotion
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown"
