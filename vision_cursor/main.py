#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import threading
import logging
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Modülleri içe aktar
from modules.eye_tracker import EyeTracker
from modules.speech_recognizer import SpeechRecognizer
from modules.gui import VisionCursorGUI

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_cursor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class WorkerThread(QThread):
    """Arka plan işlemleri için worker thread"""
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Thread hatası: {str(e)}")

def check_dependencies():
    """Gerekli bağımlılıkları kontrol et"""
    try:
        import cv2
        import mediapipe
        import PyQt5
        import pyautogui
        import speech_recognition
        import whisper
        import numpy
        import dlib
        import PIL
        return True
    except ImportError as e:
        logger.error(f"Bağımlılık hatası: {str(e)}")
        return False

def main():
    """Ana uygulama fonksiyonu"""
    
    # Bağımlılıkları kontrol et
    if not check_dependencies():
        QMessageBox.critical(None, "Hata", "Gerekli bağımlılıklar eksik. Lütfen requirements.txt dosyasındaki paketleri yükleyin.")
        sys.exit(1)
    
    # PyQt uygulamasını başlat
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Daha modern bir görünüm
    
    # High DPI desteği
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Stil ayarları için style sheet
    style_sheet = """
    QMainWindow, QWidget {
        background-color: #2d2d2d;
        color: #f0f0f0;
    }
    
    QPushButton {
        background-color: #0078d7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #0063b1;
    }
    
    QPushButton:pressed {
        background-color: #004c8c;
    }
    
    QPushButton:checked {
        background-color: #00a651;
    }
    
    QGroupBox {
        border: 2px solid #3a3a3a;
        border-radius: 5px;
        margin-top: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    
    QTextEdit {
        background-color: #3a3a3a;
        color: #f0f0f0;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 5px;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    QStatusBar {
        background-color: #222;
        color: #f0f0f0;
        border-top: 1px solid #555;
    }
    """
    
    app.setStyleSheet(style_sheet)
    
    # GUI oluştur
    main_window = VisionCursorGUI()
    
    # Modülleri başlat
    try:
        # Göz takibi modülünü oluştur
        eye_tracker = EyeTracker()
        main_window.set_eye_tracker(eye_tracker)
        
        # Ses tanıma modülünü oluştur
        speech_recognizer = SpeechRecognizer(callback=main_window.on_speech_recognized)
        main_window.set_speech_recognizer(speech_recognizer)
        
        # Modülleri ayrı thread'lerde başlat
        eye_tracker_thread = WorkerThread(eye_tracker.start)
        speech_thread = WorkerThread(speech_recognizer.start)
        
        eye_tracker_thread.error.connect(lambda msg: main_window.status_bar.showMessage(f"Göz takibi hatası: {msg}"))
        speech_thread.error.connect(lambda msg: main_window.status_bar.showMessage(f"Ses tanıma hatası: {msg}"))
        
        eye_tracker_thread.start()
        speech_thread.start()
        
        # Durumu güncelle
        main_window.status_bar.showMessage("Sistem hazır")
        logger.info("Sistem başarıyla başlatıldı")
        
    except Exception as e:
        error_msg = f"Başlatma hatası: {str(e)}"
        logger.error(error_msg)
        main_window.status_bar.showMessage(error_msg)
        QMessageBox.critical(main_window, "Hata", error_msg)
    
    # Ana pencereyi göster
    main_window.show()
    
    # Uygulama döngüsünü başlat
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 