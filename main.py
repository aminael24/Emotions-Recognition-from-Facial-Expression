import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detector import EmotionDetector, EMOTION_COLORS
from speaker import Speaker


MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")

# Nombre de frames consécutives nécessaires avant de parler.
# 2 = plus rapide pour tester.
STABLE_FRAMES = 2

# Temps minimum entre deux annonces vocales, en secondes.
SPEAK_COOLDOWN = 3


def run():
    detector = EmotionDetector(MODEL_PATH)
    speaker = Speaker()

    # Test audio au démarrage
    print("Testing speaker...")
    speaker.say("Application started")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open webcam device 0")
        sys.exit(1)

    print("Emotion Recognition — press 'q' to quit, 's' to save screenshot")

    screenshot_idx = 0

    announced_emotion = None
    candidate = None
    candidate_count = 0
    last_speak_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Réduire l'image pour accélérer la détection des visages
        gray_small = cv2.resize(gray, None, fx=0.5, fy=0.5)

        faces_small = detector.detect_faces(gray_small)

        # Remettre les coordonnées à la taille originale
        faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces_small]

        # Prendre le plus grand visage comme visage principal
        primary = max(faces, key=lambda f: f[2] * f[3]) if len(faces) > 0 else None
        primary_emotion = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            emotion, confidence = detector.predict(face_roi)
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))

            if primary is not None and (x, y, w, h) == primary:
                primary_emotion = emotion

            # Dessiner rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Afficher émotion + confiance
            label = f"{emotion} {confidence:.0%}"
            label_y = y - 12 if y - 12 > 12 else y + h + 20

            cv2.putText(
                frame,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA
            )

        # Debug : afficher l'émotion principale détectée
        if primary_emotion is not None:
            print("Detected emotion:", primary_emotion)

        # Stabilisation de l'émotion
        if primary_emotion == candidate:
            candidate_count += 1
        else:
            candidate = primary_emotion
            candidate_count = 1

        # Annonce vocale
        current_time = time.time()

        if (
            candidate is not None
            and candidate_count >= STABLE_FRAMES
            and candidate != announced_emotion
            and current_time - last_speak_time >= SPEAK_COOLDOWN
        ):
            announced_emotion = candidate
            last_speak_time = current_time

            text = f"This person is {candidate}"
            print("Speaking:", text)
            speaker.say(text)

        # Informations affichées sur la fenêtre
        info = f"Faces detected: {len(faces)} | q=quit | s=screenshot"

        cv2.putText(
            frame,
            info,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            fname = f"screenshot_{screenshot_idx:03d}.png"

            if cv2.imwrite(fname, frame):
                print(f"Screenshot saved: {fname}")
            else:
                print(f"Error: could not save {fname}")

            screenshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()