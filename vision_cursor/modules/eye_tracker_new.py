import numpy as np
import mediapipe as mp
import pyautogui
import time
import threading
import cv2
from .camera import Camera

class EyeTracker:
    def __init__(self):
        # MediaPipe yüz algılama modülü
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Ekran boyutları
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Kamera
        self.camera = Camera(width=640, height=480, fps=30)
        
        # İzleme durumu
        self.tracking = False
        
        # Göz takibi için değişkenler
        self.last_positions = []
        self.smooth_factor = 8
        self.frame_skip = 2
        self.frame_count = 0
        
        # Tıklama için değişkenler
        self.gaze_duration = 0
        self.last_gaze_pos = None
        self.gaze_threshold = 1.5
        self.movement_threshold = 15
        self.last_time = time.time()
        
        # Kalibrasyon
        self.offset_x = 0
        self.offset_y = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Göz landmark indeksleri (MediaPipe)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
    def start(self):
        if self.tracking:
            return
            
        try:
            if not self.camera.start(callback=self._process_frame):
                raise ValueError("Kamera başlatılamadı!")
                
            self.tracking = True
            print("Göz takibi başlatıldı")
            
        except Exception as e:
            self.camera.stop()
            raise ValueError(f"Göz takibi başlatma hatası: {str(e)}")
    
    def stop(self):
        self.tracking = False
        self.camera.stop()
        print("Göz takibi durduruldu")
    
    def _process_frame(self, image):
        if not self.tracking:
            return
            
        # Performans için frame atlama
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
            
        try:
            # RGB'ye çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe ile yüz tespiti
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # İris merkezlerini al
                left_iris_center = self._get_iris_center(face_landmarks, self.LEFT_IRIS)
                right_iris_center = self._get_iris_center(face_landmarks, self.RIGHT_IRIS)
                
                if left_iris_center and right_iris_center:
                    # Ortalamasını al
                    avg_x = (left_iris_center[0] + right_iris_center[0]) / 2
                    avg_y = (left_iris_center[1] + right_iris_center[1]) / 2
                    
                    # Görüntü koordinatlarına çevir
                    img_x = int(avg_x * image.shape[1])
                    img_y = int(avg_y * image.shape[0])
                    
                    # Pozisyonu yumuşat
                    smooth_x, smooth_y = self._smooth_position(img_x, img_y)
                    
                    # Ekran koordinatlarına ölçekle
                    screen_x = self._map_to_screen_x(smooth_x, image.shape[1])
                    screen_y = self._map_to_screen_y(smooth_y, image.shape[0])
                    
                    # İmleci hareket ettir
                    pyautogui.moveTo(int(screen_x), int(screen_y))
                    
                    # Tıklama kontrolü
                    self._check_for_click(smooth_x, smooth_y)
                    
                    # Görselleştirme
                    self._draw_tracking_info(image, img_x, img_y)
            
        except Exception as e:
            print(f"Frame işleme hatası: {str(e)}")
    
    def _get_iris_center(self, face_landmarks, iris_indices):
        """İris merkezini hesapla"""
        try:
            x_coords = []
            y_coords = []
            
            for idx in iris_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
            
            if x_coords and y_coords:
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                return (center_x, center_y)
            
            return None
        except Exception as e:
            print(f"İris merkezi hesaplama hatası: {e}")
            return None
    
    def _smooth_position(self, x, y):
        """Pozisyonu yumuşat"""
        self.last_positions.append((x, y))
        if len(self.last_positions) > self.smooth_factor:
            self.last_positions.pop(0)
        
        if self.last_positions:
            avg_x = sum(p[0] for p in self.last_positions) / len(self.last_positions)
            avg_y = sum(p[1] for p in self.last_positions) / len(self.last_positions)
            return avg_x, avg_y
        
        return x, y
    
    def _map_to_screen_x(self, img_x, img_width):
        """X koordinatını ekrana ölçekle"""
        normalized_x = img_x / img_width
        screen_x = normalized_x * self.screen_width * self.scale_x + self.offset_x
        return max(0, min(self.screen_width - 1, screen_x))
    
    def _map_to_screen_y(self, img_y, img_height):
        """Y koordinatını ekrana ölçekle"""
        normalized_y = img_y / img_height
        screen_y = normalized_y * self.screen_height * self.scale_y + self.offset_y
        return max(0, min(self.screen_height - 1, screen_y))
    
    def _check_for_click(self, x, y):
        """Sabit bakış tıklama kontrolü"""
        try:
            current_time = time.time()
            
            if self.last_gaze_pos is not None:
                # Hareket mesafesini hesapla
                movement = np.sqrt((x - self.last_gaze_pos[0])**2 + 
                                 (y - self.last_gaze_pos[1])**2)
                
                if movement < self.movement_threshold:
                    # Sabit bakış süresi arttır
                    self.gaze_duration += (current_time - self.last_time)
                    
                    if self.gaze_duration > self.gaze_threshold:
                        # Tıklama yap
                        pyautogui.click()
                        self.gaze_duration = 0
                        print("Göz tıklaması!")
                else:
                    # Hareket varsa süreyi sıfırla
                    self.gaze_duration = 0
            
            self.last_gaze_pos = (x, y)
            self.last_time = current_time
            
        except Exception as e:
            print(f"Tıklama kontrolü hatası: {e}")
    
    def _draw_tracking_info(self, image, x, y):
        """Takip bilgilerini çiz"""
        try:
            # Göz pozisyonunu işaretle
            cv2.circle(image, (int(x), int(y)), 5, (255, 192, 203), -1)
            
            # Tıklama progress'ini göster
            if self.gaze_duration > 0:
                progress = min(1.0, self.gaze_duration / self.gaze_threshold)
                radius = int(30 * progress)
                cv2.circle(image, (int(x), int(y)), radius, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
    
    def calibrate(self, offset_x=0, offset_y=0, scale_x=1.0, scale_y=1.0):
        """Kalibrasyon ayarları"""
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.scale_x = scale_x
        self.scale_y = scale_y
        print(f"Kalibrasyon: offset({offset_x}, {offset_y}), scale({scale_x}, {scale_y})")
    
    def get_frame(self):
        """Mevcut frame'i al"""
        return self.camera.get_frame()
