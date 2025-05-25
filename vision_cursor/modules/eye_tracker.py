import numpy as np
import mediapipe as mp
import pyautogui
import time
import threading
from .camera import Camera
from PIL import Image, ImageDraw

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
        self.camera = Camera(width=640, height=480, fps=30)
        
        # İzleme durumu
        self.tracking = False
        
        # Göz bebeği takibi için değişkenler
        self.pupil_threshold = 30
        self.pupil_radius = 5
        self.pupil_color = (255, 192, 203)  # Pembe renk
        
        # Göz bebeği pozisyonu için değişkenler
        self.last_pupil_pos = None
        self.pupil_movement_threshold = 5
        
        # Hareket yumuşatma için değişkenler
        self.last_positions = []
        self.smooth_factor = 5
        
        # Tıklama için değişkenler
        self.gaze_duration = 0
        self.last_gaze_pos = (0, 0)
        self.gaze_threshold = 2.0
        self.gaze_radius = 30
        
        # Göz referans noktaları (MediaPipe indeksleri)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Kalibrasyon için offset
        self.offset_x = 0
        self.offset_y = 0

    def start(self):
        if self.tracking:
            return
            
        try:
            # Kamera başlat
            if not self.camera.start(callback=self._process_frame):
                raise ValueError("Kamera başlatılamadı!")
                
            self.tracking = True
            
        except Exception as e:
            self.camera.stop()
            raise ValueError(f"Göz takibi başlatma hatası: {str(e)}")
    
    def stop(self):
        self.tracking = False
        self.camera.stop()
    
    def _process_frame(self, image):
        if not self.tracking:
            return
            
        try:
            # MediaPipe ile yüz tespiti
            results = self.face_mesh.process(image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Sol ve sağ göz bölgelerini al
                left_eye_region = self._get_eye_region(image, face_landmarks, self.LEFT_EYE)
                right_eye_region = self._get_eye_region(image, face_landmarks, self.RIGHT_EYE)
                
                if left_eye_region is not None and right_eye_region is not None:
                    # Göz bebeklerini tespit et
                    left_pupil = self._detect_pupil(left_eye_region)
                    right_pupil = self._detect_pupil(right_eye_region)
                    
                    if left_pupil is not None and right_pupil is not None:
                        # Göz bebeği merkezlerini global koordinata çevir
                        left_global = self._eye_local_to_global(image, face_landmarks, self.LEFT_EYE, left_pupil)
                        right_global = self._eye_local_to_global(image, face_landmarks, self.RIGHT_EYE, right_pupil)
                        # Ortalamasını al
                        avg_x = int((left_global[0] + right_global[0]) / 2)
                        avg_y = int((left_global[1] + right_global[1]) / 2)
                        # Ekran koordinatına ölçekle
                        screen_x = np.interp(avg_x, [0, image.shape[1]], [0, self.screen_width]) + self.offset_x
                        screen_y = np.interp(avg_y, [0, image.shape[0]], [0, self.screen_height]) + self.offset_y
                        # İmleci doğrudan götür
                        pyautogui.moveTo(int(screen_x), int(screen_y))
                        
                        current_time = time.time()
                        
                        # Göz bebeği hareketini kontrol et
                        if self.last_pupil_pos is not None:
                            movement = np.sqrt((avg_x - self.last_pupil_pos[0])**2 + 
                                             (avg_y - self.last_pupil_pos[1])**2)
                            
                            if movement < self.pupil_movement_threshold:
                                self.gaze_duration += time.time() - current_time
                                
                                if self.gaze_duration > self.gaze_threshold:
                                    pyautogui.click()
                                    self.gaze_duration = 0
                            else:
                                self.gaze_duration = 0
                        
                        self.last_pupil_pos = (avg_x, avg_y)
                        
                        # Görselleştirme
                        self._draw_pupils(image, left_global, right_global)
            
        except Exception as e:
            print(f"Görüntü işleme hatası: {str(e)}")
    
    def _detect_pupil(self, eye_region):
        try:
            # Gri tonlamaya çevir
            gray = np.mean(eye_region, axis=2).astype(np.uint8)
            
            # Gürültüyü azalt
            blurred = self._gaussian_blur(gray, kernel_size=7)
            
            # Eşikleme
            thresh = self._threshold(blurred, self.pupil_threshold)
            
            # Konturları bul
            contours = self._find_contours(thresh)
            
            if contours:
                # En büyük konturu al (göz bebeği)
                largest_contour = max(contours, key=self._contour_area)
                
                # Göz bebeğinin merkezini hesapla
                cx, cy = self._contour_center(largest_contour)
                return (int(cx), int(cy))
            
            return None
        except Exception as e:
            print(f"Göz bebeği tespiti hatası: {str(e)}")
            return None
    
    def _get_eye_region(self, image, face_landmarks, eye_indices):
        points = []
        for idx in eye_indices:
            point = face_landmarks.landmark[idx]
            points.append([point.x, point.y])
        
        if points:
            x_min = int(min(p[0] for p in points) * image.shape[1])
            x_max = int(max(p[0] for p in points) * image.shape[1])
            y_min = int(min(p[1] for p in points) * image.shape[0])
            y_max = int(max(p[1] for p in points) * image.shape[0])
            
            return image[y_min:y_max, x_min:x_max]
        return None
    
    def _eye_local_to_global(self, image, face_landmarks, eye_indices, pupil):
        points = []
        for idx in eye_indices:
            point = face_landmarks.landmark[idx]
            points.append([point.x, point.y])
        
        if points:
            x_min = int(min(p[0] for p in points) * image.shape[1])
            y_min = int(min(p[1] for p in points) * image.shape[0])
            return (x_min + pupil[0], y_min + pupil[1])
        return (0, 0)
    
    def _smooth_position(self, x, y):
        self.last_positions.append((x, y))
        if len(self.last_positions) > self.smooth_factor:
            self.last_positions.pop(0)
        
        avg_x = np.mean([p[0] for p in self.last_positions])
        avg_y = np.mean([p[1] for p in self.last_positions])
        
        return avg_x, avg_y
        
    def _gaussian_blur(self, image, kernel_size=7):
        """Basit Gaussian blur uygular."""
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return self._convolve2d(image, kernel)
        
    def _convolve2d(self, image, kernel):
        """2D konvolüsyon uygular."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
                
        return output.astype(np.uint8)
        
    def _threshold(self, image, threshold):
        """Basit eşikleme uygular."""
        return (image < threshold).astype(np.uint8) * 255
        
    def _find_contours(self, image):
        """Basit kontur bulma algoritması."""
        h, w = image.shape
        visited = np.zeros_like(image, dtype=bool)
        contours = []
        
        def flood_fill(x, y, contour):
            if (x < 0 or x >= w or y < 0 or y >= h or 
                visited[y, x] or image[y, x] == 0):
                return
                
            visited[y, x] = True
            contour.append((x, y))
            
            flood_fill(x+1, y, contour)
            flood_fill(x-1, y, contour)
            flood_fill(x, y+1, contour)
            flood_fill(x, y-1, contour)
        
        for y in range(h):
            for x in range(w):
                if not visited[y, x] and image[y, x] > 0:
                    contour = []
                    flood_fill(x, y, contour)
                    if len(contour) > 10:  # Minimum kontur boyutu
                        contours.append(np.array(contour))
        
        return contours
        
    def _contour_area(self, contour):
        """Kontur alanını hesaplar."""
        return len(contour)
        
    def _contour_center(self, contour):
        """Kontur merkezini hesaplar."""
        return np.mean(contour, axis=0)
        
    def _draw_pupils(self, image, left_pupil, right_pupil):
        """Göz bebeklerini görselleştirir."""
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Sol göz bebeği
        draw.ellipse([
            left_pupil[0] - self.pupil_radius,
            left_pupil[1] - self.pupil_radius,
            left_pupil[0] + self.pupil_radius,
            left_pupil[1] + self.pupil_radius
        ], fill=self.pupil_color)
        
        # Sağ göz bebeği
        draw.ellipse([
            right_pupil[0] - self.pupil_radius,
            right_pupil[1] - self.pupil_radius,
            right_pupil[0] + self.pupil_radius,
            right_pupil[1] + self.pupil_radius
        ], fill=self.pupil_color)
        
        # PIL görüntüsünü numpy dizisine dönüştür
        image[:] = np.array(pil_image)
    
    def get_frame(self):
        return self.camera.get_frame() 