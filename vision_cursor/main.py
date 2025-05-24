#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from modules.gui import VisionCursorGUI
from modules.eye_tracker import EyeTracker
from modules.speech_recognizer import SpeechRecognizer
import threading

# Loglama yapılandırması
logging.basicConfig(
    filename='vision_cursor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WorkerThread(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.daemon = True

    def run(self):
        try:
            self.target(*self.args, **self.kwargs)
        except Exception as e:
            logging.error(f"Thread hatası: {str(e)}")

def check_dependencies():
    """Gerekli paketlerin yüklü olup olmadığını kontrol et"""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('PyQt5', 'PyQt5'),
        ('pyautogui', 'pyautogui'),
        ('speech_recognition', 'SpeechRecognition'),
        ('whisper', 'openai-whisper'),
        ('numpy', 'numpy'),
        ('face_recognition', 'face-recognition'),
        ('PIL', 'pillow'),
        ('dotenv', 'python-dotenv')
    ]
    missing_packages = []
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    if missing_packages:
        error_msg = f"Eksik paketler: {', '.join(missing_packages)}"
        logging.error(error_msg)
        return False, error_msg
    return True, None

def main():
    try:
        # PyQt uygulamasını başlat
        app = QApplication(sys.argv)
        
        # Modern stil ayarla
        app.setStyle('Fusion')
        
        # Bağımlılıkları kontrol et
        deps_ok, error_msg = check_dependencies()
        if not deps_ok:
            QMessageBox.critical(None, "Hata", f"Gerekli paketler eksik!\n\n{error_msg}\n\nLütfen requirements.txt dosyasındaki paketleri yükleyin.")
            return
        
        # Ana pencereyi oluştur
        window = VisionCursorGUI()
        window.show()
        
        # Göz takibi ve ses tanıma modüllerini başlat
        try:
            eye_tracker = EyeTracker()
            speech_recognizer = SpeechRecognizer(callback=window.on_speech_recognized)
            
            # Modülleri GUI'ye bağla
            window.set_eye_tracker(eye_tracker)
            window.set_speech_recognizer(speech_recognizer)
            
            # Göz takibini başlat
            eye_tracker.start()
            window.eye_tracking_active = True
            window.eye_tracking_button.setText("Göz Takibini Durdur")
            window.status_bar.showMessage("Göz takibi aktif")
            logging.info("Göz takibi başlatıldı")
            
            # Ses tanımayı başlat
            speech_recognizer.start()
            window.speech_recognition_active = True
            window.speech_button.setText("Ses Tanımayı Durdur")
            window.status_bar.showMessage("Ses tanıma aktif")
            logging.info("Ses tanıma başlatıldı")
            
        except Exception as e:
            error_msg = f"Modül başlatma hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(window, "Hata", error_msg)
            return
        
        # Uygulama döngüsünü başlat
        sys.exit(app.exec_())
        
    except Exception as e:
        error_msg = f"Uygulama hatası: {str(e)}"
        logging.error(error_msg)
        QMessageBox.critical(None, "Hata", error_msg)
        raise

if __name__ == "__main__":
    main() 