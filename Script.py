
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import pickle
import time
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class OptimizedFaceAnalyzer:
    def __init__(self, 
                 emotion_model_path='emotion_custom_model.keras',
                 label_encoder_path='label_encoder.pkl'):
        print("\n" + "="*60)
        print("LOADING MODELS (Optimized)")
        print("="*60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("[1/4] Face detector...", end=' ')
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓")
        print("[2/4] Emotion model...", end=' ')
        self.emotion_model = keras.models.load_model(emotion_model_path, compile=False)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓")
        print("[3/4] Gender model...", end=' ')
        self.gender_extractor = AutoFeatureExtractor.from_pretrained(
            "rizvandwiki/gender-classification-2"
        )
        self.gender_model = AutoModelForImageClassification.from_pretrained(
            "rizvandwiki/gender-classification-2"
        ).to(self.device).eval()
        print("✓")

        print("[4/4] Age model...", end=' ')
        self.age_processor = ViTImageProcessor.from_pretrained(
            "nateraw/vit-age-classifier"
        )
        self.age_model = ViTForImageClassification.from_pretrained(
            "nateraw/vit-age-classifier"
        ).to(self.device).eval()
        print("✓")
        
        print("="*60)
        print("✓ ALL MODELS LOADED")
        print("="*60 + "\n")
    
    @torch.no_grad()
    def predict_all(self, face_img):
        
        results = {}

        try:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_face, (48, 48))
            normalized = resized.astype('float32') / 255.0
            input_img = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
            
            predictions = self.emotion_model.predict(input_img, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion_conf = float(predictions[0][emotion_idx])
            emotion_label = self.label_encoder.classes_[emotion_idx]
            
            results['emotion'] = (emotion_label, emotion_conf)
        except:
            results['emotion'] = (None, None)

        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
        
            gender_inputs = self.gender_extractor(images=pil_img, return_tensors="pt")
            gender_inputs = {k: v.to(self.device) for k, v in gender_inputs.items()}
            
            gender_logits = self.gender_model(**gender_inputs).logits
            gender_idx = gender_logits.argmax(-1).item()
            gender_probs = torch.nn.functional.softmax(gender_logits, dim=-1)
            gender_conf = float(gender_probs[0][gender_idx])
            gender_label = self.gender_model.config.id2label[gender_idx]
            
            results['gender'] = (gender_label, gender_conf)
            
           
            age_inputs = self.age_processor(pil_img, return_tensors='pt')
            age_inputs = {k: v.to(self.device) for k, v in age_inputs.items()}
            
            age_output = self.age_model(**age_inputs)
            age_probs = age_output.logits.softmax(1)
            age_idx = age_probs.argmax(1).item()
            age_conf = float(age_probs[0][age_idx])
            age_label = self.age_model.config.id2label[age_idx]
            
            results['age'] = (age_label, age_conf)
        except:
            results['gender'] = (None, None)
            results['age'] = (None, None)
        
        return results
def run_ultra_light(
    emotion_model_path='emotion_custom_model.keras',
    label_encoder_path='label_encoder.pkl',
    scale_factor=1.3,
    min_neighbors=5,
    process_every_n_frames=4,  
    resize_factor=0.75,  
    max_faces=3  
):
    analyzer = OptimizedFaceAnalyzer(emotion_model_path, label_encoder_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("\n" + "="*60)
    print("WEBCAM STARTED - Ultra-Light Mode")
    print("="*60)
    print("Controls:")
    print("  Q - Quit")
    print("  SPACE - Toggle all predictions")
    print("  F - Toggle FPS")
    print("  + - Increase processing frequency")
    print("  - - Decrease processing frequency")
    print("="*60 + "\n")
    frame_count = 0
    cached_results = {}
    show_predictions = True
    show_fps = True
    fps_deque = deque(maxlen=30)
    fps_time = time.time()
    current_skip = process_every_n_frames 
    while True:
        loop_start = time.time()  
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces_small = analyzer.face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(20, 20)
        )
        scale_back = 1.0 / resize_factor
        faces = [(int(x * scale_back), int(y * scale_back), 
                  int(w * scale_back), int(h * scale_back)) 
                 for x, y, w, h in faces_small[:max_faces]]
        if frame_count % current_skip == 0 and show_predictions:
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face
                padding = 10
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    results = analyzer.predict_all(face_img)
                    cached_results[i] = results
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if show_predictions and i in cached_results:
                results = cached_results[i]
                if results['emotion'][0]:
                    emotion, conf = results['emotion']
                    text = f"{emotion} {conf:.0%}"
                    cv2.putText(frame, text, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if results['gender'][0]:
                    gender, conf = results['gender']
                    text = f"{gender} {conf:.0%}"
                    cv2.putText(frame, text, (x, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                if results['age'][0]:
                    age, conf = results['age']
                    text = f"{age} {conf:.0%}"
                    cv2.putText(frame, text, (x, y + h + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        loop_time = time.time() - loop_start
        if loop_time > 0:
            fps_deque.append(1.0 / loop_time)

        if show_fps and len(fps_deque) > 0:
            fps = int(np.mean(fps_deque))
            cv2.putText(frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
        status = f"Skip: {current_skip} | Faces: {len(faces)}"
        cv2.putText(frame, status, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Analysis - Ultra Light (Press Q to quit)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            show_predictions = not show_predictions
            print(f"Predictions: {'ON' if show_predictions else 'OFF'}")
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('+') or key == ord('='):
            current_skip = max(1, current_skip - 1)
            print(f"Processing frequency increased (skip: {current_skip})")
        elif key == ord('-') or key == ord('_'):
            current_skip = min(10, current_skip + 1)
            print(f"Processing frequency decreased (skip: {current_skip})")
        
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Light Face Analysis')
    parser.add_argument('--emotion-model', type=str, 
                       default='emotion_custom_model.keras')
    parser.add_argument('--label-encoder', type=str,
                       default='label_encoder.pkl')
    parser.add_argument('--skip', type=int, default=4,
                       help='Process every Nth frame (higher = faster)')
    parser.add_argument('--resize', type=float, default=0.75,
                       help='Resize factor for detection (lower = faster)')
    parser.add_argument('--max-faces', type=int, default=3,
                       help='Maximum faces to process')
    
    args = parser.parse_args()
    
    run_ultra_light(
        emotion_model_path=args.emotion_model,
        label_encoder_path=args.label_encoder,
        process_every_n_frames=args.skip,
        resize_factor=args.resize,
        max_faces=args.max_faces
    )