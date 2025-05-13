import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading

class EyeTracker:
    def __init__(self):
        # Mediapipe yüz algılama modülünü başlat
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Ekran boyutları
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Kamera başlatma
        self.cap = None
        
        # İzleme durumu
        self.tracking = False
        self.thread = None
        
        # Göz sabitlenme süresi kontrolü
        self.gaze_duration = 0
        self.last_gaze_pos = (0, 0)
        self.gaze_threshold = 2.0  # Göz sabitlenmesi için gerekli süre (saniye)
        self.gaze_radius = 30  # Piksel cinsinden göz sabitlenmesi için kabul edilebilir sapma
        
        # Kalibrasyon için değişkenler
        self.is_calibrated = False
        self.calibration_points = []
        self.calibration_count = 0
        
        # Göz referans noktaları (MediaPipe indeksleri)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Pürüzsüzleştirme için son konumlar
        self.last_positions = []
        self.smooth_factor = 5

    def start(self):
        if self.tracking:
            return
            
        # Kamera başlat
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Kamera açılamadı!")
            
        self.tracking = True
        self.thread = threading.Thread(target=self._track_eyes)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.tracking = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
    
    def _track_eyes(self):
        last_click_time = 0
        
        while self.tracking:
            success, image = self.cap.read()
            if not success:
                continue
                
            # BGR'dan RGB'ye dönüştürme
            image = cv2.flip(image, 1)  # Ayna etkisi için yatay çevirme
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            # Mediapipe işleme
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Göz merkezlerini hesapla
                left_eye_center = self._get_eye_center(face_landmarks, self.LEFT_EYE)
                right_eye_center = self._get_eye_center(face_landmarks, self.RIGHT_EYE)
                
                if left_eye_center and right_eye_center:
                    # İki göz arasındaki orta nokta
                    eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    eye_mid_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    
                    # Normalizasyon ve ekran koordinatlarına dönüştürme
                    screen_x = np.interp(eye_mid_x, [0.0, 1.0], [0, self.screen_width])
                    screen_y = np.interp(eye_mid_y, [0.0, 1.0], [0, self.screen_height])
                    
                    # Konumu pürüzsüzleştir
                    smoothed_pos = self._smooth_position(screen_x, screen_y)
                    cursor_x, cursor_y = smoothed_pos
                    
                    # İmleci hareket ettir
                    pyautogui.moveTo(cursor_x, cursor_y)
                    
                    # Gözün sabit kalıp kalmadığını kontrol et
                    current_time = time.time()
                    
                    # Önceki pozisyonla şimdiki pozisyon arasındaki fark
                    distance = np.sqrt((cursor_x - self.last_gaze_pos[0])**2 + 
                                      (cursor_y - self.last_gaze_pos[1])**2)
                    
                    if distance < self.gaze_radius:
                        # Aynı bölgede ise süreyi artır
                        self.gaze_duration += time.time() - current_time
                        
                        # Gözler belirli bir süre sabit kaldıysa tıkla
                        if self.gaze_duration > self.gaze_threshold and current_time - last_click_time > 1.0:
                            pyautogui.click()
                            self.gaze_duration = 0
                            last_click_time = current_time
                    else:
                        # Göz pozisyonu değişti, sabit kalma süresini sıfırla
                        self.gaze_duration = 0
                        self.last_gaze_pos = (cursor_x, cursor_y)
                    
            # Görüntüyü işleme ve görselleştirme buraya eklenebilir
            
    def _get_eye_center(self, landmarks, eye_indices):
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y])
        
        if points:
            center = np.mean(points, axis=0)
            return center
        return None
    
    def _smooth_position(self, x, y):
        self.last_positions.append((x, y))
        if len(self.last_positions) > self.smooth_factor:
            self.last_positions.pop(0)
        
        avg_x = np.mean([p[0] for p in self.last_positions])
        avg_y = np.mean([p[1] for p in self.last_positions])
        
        return avg_x, avg_y
    
    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
            
        success, image = self.cap.read()
        if not success:
            return None
            
        # Görüntüyü aynalama
        image = cv2.flip(image, 1)
        
        # Mediapipe ile işleme
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        # Yüz noktalarını çiz
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Göz noktalarını çiz
                for idx in self.LEFT_EYE + self.RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    h, w, _ = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                
                # Göz merkezlerini çiz
                left_eye_center = self._get_eye_center(face_landmarks, self.LEFT_EYE)
                right_eye_center = self._get_eye_center(face_landmarks, self.RIGHT_EYE)
                
                if left_eye_center and right_eye_center:
                    h, w, _ = image.shape
                    left_x, left_y = int(left_eye_center[0] * w), int(left_eye_center[1] * h)
                    right_x, right_y = int(right_eye_center[0] * w), int(right_eye_center[1] * h)
                    
                    cv2.circle(image, (left_x, left_y), 5, (255, 0, 0), -1)
                    cv2.circle(image, (right_x, right_y), 5, (255, 0, 0), -1)
        
        return image 