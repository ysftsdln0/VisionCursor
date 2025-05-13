#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Modülleri içe aktar
from modules.eye_tracker import EyeTracker
from modules.speech_recognizer import SpeechRecognizer
from modules.gui import VisionCursorGUI

def main():
    """Ana uygulama fonksiyonu"""
    
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
        
        # Durumu güncelle
        main_window.status_bar.showMessage("Sistem hazır")
        
    except Exception as e:
        main_window.status_bar.showMessage(f"Hata: {str(e)}")
        print(f"Başlatma hatası: {str(e)}")
    
    # Ana pencereyi göster
    main_window.show()
    
    # Uygulama döngüsünü başlat
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 