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
from modules.performance_monitor import PerformanceMonitor
import threading

# log ayarları
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
            logging.error(f"thread hatası: {str(e)}")

def check_dependencies():
    """paketlerin olup olmadığını kontrol et"""
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
        error_msg = f"eksik paketler: {', '.join(missing_packages)}"
        logging.error(error_msg)
        return False, error_msg
    return True, None

def main():
    try:
        # pyqt uygulamasını başlat
        app = QApplication(sys.argv)
        
        # stil ayarla
        app.setStyle('Fusion')
        
        # paketleri kontrol et
        deps_ok, error_msg = check_dependencies()
        if not deps_ok:
            QMessageBox.critical(None, "hata", f"paketler eksik!\n\n{error_msg}\n\nrequirements.txt dosyasındaki paketleri yükle.")
            return
        
        # ana pencereyi oluştur
        window = VisionCursorGUI()
        window.show()
        
        # göz takibi ve ses modüllerini başlat
        try:
            # Performans monitörü başlat
            performance_monitor = PerformanceMonitor()
            performance_monitor.start_monitoring()
            
            eye_tracker = EyeTracker()
            speech_recognizer = SpeechRecognizer(callback=window.on_speech_recognized)
            
            # modülleri gui'ye bağla
            window.set_eye_tracker(eye_tracker)
            window.set_speech_recognizer(speech_recognizer)
            window.set_performance_monitor(performance_monitor)
            
            # göz takibini başlat
            eye_tracker.start()
            window.eye_tracking_active = True
            window.eye_tracking_button.setText("göz takibini durdur")
            window.status_bar.showMessage("göz takibi aktif")
            logging.info("göz takibi başlatıldı")
            
            # ses tanıma kullanıcı başlatabilir
            window.speech_recognition_active = False
            window.speech_button.setText("ses tanımayı başlat")
            window.status_bar.showMessage("ses tanıma hazır")
            logging.info("ses tanıma hazır")
            
        except Exception as e:
            error_msg = f"modül başlatma hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(window, "hata", error_msg)
            return
        
        # uygulama döngüsünü başlat
        sys.exit(app.exec_())
        
    except Exception as e:
        error_msg = f"Uygulama hatası: {str(e)}"
        logging.error(error_msg)
        QMessageBox.critical(None, "Hata", error_msg)
        raise

if __name__ == "__main__":
    main() 