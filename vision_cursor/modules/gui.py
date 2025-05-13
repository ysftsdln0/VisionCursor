import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QGroupBox, QSplitter, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
import cv2
import numpy as np
import os

class VisionCursorGUI(QMainWindow):
    # Özel sinyaller
    speech_command_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Pencere ayarları
        self.setWindowTitle("VisionCursor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Pencere oluşturma
        self._create_ui()
        
        # Kamera ve Ses Modülleri için referanslar
        self.eye_tracker = None
        self.speech_recognizer = None
        
        # Kamera güncelleme zamanlayıcısı
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(30)  # 30ms (yaklaşık 33 FPS)
        
        # Başlangıçta modüllerin durumu
        self.eye_tracking_active = False
        self.speech_recognition_active = False
        
    def _create_ui(self):
        """UI bileşenlerini oluşturur"""
        # Ana düzen için yatay bölücü
        splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter)
        
        # Sol panel (kamera ve kontroller)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Kamera görüntüsü
        self.camera_group = QGroupBox("Kamera Görüntüsü")
        camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #333; border-radius: 10px;")
        camera_layout.addWidget(self.camera_label)
        self.camera_group.setLayout(camera_layout)
        left_layout.addWidget(self.camera_group)
        
        # Kontrol butonları
        control_layout = QHBoxLayout()
        
        # Göz takibi kontrolleri
        self.eye_tracking_button = QPushButton("Göz Takibini Başlat")
        self.eye_tracking_button.setCheckable(True)
        self.eye_tracking_button.setFont(QFont("Arial", 14))
        self.eye_tracking_button.setMinimumHeight(60)
        self.eye_tracking_button.clicked.connect(self.toggle_eye_tracking)
        control_layout.addWidget(self.eye_tracking_button)
        
        # Ses tanıma kontrolleri
        self.speech_button = QPushButton("Ses Tanımayı Başlat")
        self.speech_button.setCheckable(True)
        self.speech_button.setFont(QFont("Arial", 14))
        self.speech_button.setMinimumHeight(60)
        self.speech_button.clicked.connect(self.toggle_speech_recognition)
        control_layout.addWidget(self.speech_button)
        
        left_layout.addLayout(control_layout)
        
        # Kaydetme butonu
        self.save_button = QPushButton("Metni Kaydet")
        self.save_button.setFont(QFont("Arial", 14))
        self.save_button.setMinimumHeight(60)
        self.save_button.clicked.connect(self.save_text)
        left_layout.addWidget(self.save_button)
        
        # Sağ panel (metin kutusu)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Metin kutusu
        text_group = QGroupBox("Tanınan Metin")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Arial", 16))
        self.text_edit.setReadOnly(False)  # Kullanıcı düzenleyebilsin
        text_layout.addWidget(self.text_edit)
        text_group.setLayout(text_layout)
        right_layout.addWidget(text_group)
        
        # Temizle butonu
        self.clear_button = QPushButton("Metni Temizle")
        self.clear_button.setFont(QFont("Arial", 14))
        self.clear_button.setMinimumHeight(60)
        self.clear_button.clicked.connect(self.clear_text)
        right_layout.addWidget(self.clear_button)
        
        # Bölücüye panelleri ekle
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])  # Başlangıç boyutu
        
        # Durum çubuğu
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Hazır")
        
    def set_eye_tracker(self, eye_tracker):
        """Göz takip modülünü ayarlar"""
        self.eye_tracker = eye_tracker
        
    def set_speech_recognizer(self, speech_recognizer):
        """Ses tanıma modülünü ayarlar"""
        self.speech_recognizer = speech_recognizer
        
    def update_camera_feed(self):
        """Kamera görüntüsünü günceller"""
        if self.eye_tracker and self.eye_tracking_active:
            frame = self.eye_tracker.get_frame()
            if frame is not None:
                # OpenCV BGR formatından Qt için RGB formatına dönüştürme
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.width(), self.camera_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def toggle_eye_tracking(self):
        """Göz takibini açıp/kapatır"""
        if self.eye_tracker:
            if not self.eye_tracking_active:
                # Başlat
                try:
                    self.eye_tracker.start()
                    self.eye_tracking_active = True
                    self.eye_tracking_button.setText("Göz Takibini Durdur")
                    self.status_bar.showMessage("Göz takibi aktif")
                except Exception as e:
                    self.status_bar.showMessage(f"Hata: {str(e)}")
            else:
                # Durdur
                self.eye_tracker.stop()
                self.eye_tracking_active = False
                self.eye_tracking_button.setText("Göz Takibini Başlat")
                self.status_bar.showMessage("Göz takibi durduruldu")
                self.camera_label.clear()
                self.camera_label.setText("Kamera kapalı")
    
    def toggle_speech_recognition(self):
        """Ses tanımayı açıp/kapatır"""
        if self.speech_recognizer:
            if not self.speech_recognition_active:
                # Başlat
                self.speech_recognizer.callback = self.on_speech_recognized
                self.speech_recognizer.start()
                self.speech_recognition_active = True
                self.speech_button.setText("Ses Tanımayı Durdur")
                self.status_bar.showMessage("Ses tanıma aktif")
            else:
                # Durdur
                self.speech_recognizer.stop()
                self.speech_recognition_active = False
                self.speech_button.setText("Ses Tanımayı Başlat")
                self.status_bar.showMessage("Ses tanıma durduruldu")
    
    def on_speech_recognized(self, text, is_command=False, command=None):
        """Tanınan metin için callback"""
        if is_command:
            # Komut işleme
            self.speech_command_signal.emit(command)
            self.status_bar.showMessage(f"Komut algılandı: {command}")
            
            # Temel komutlar
            if command == "clear":
                self.clear_text()
            elif command == "save":
                self.save_text()
            elif command == "start_eye":
                if not self.eye_tracking_active:
                    self.toggle_eye_tracking()
            elif command == "stop_eye":
                if self.eye_tracking_active:
                    self.toggle_eye_tracking()
            elif command == "start_speech":
                if not self.speech_recognition_active:
                    self.toggle_speech_recognition()
            elif command == "stop_speech":
                if self.speech_recognition_active:
                    self.toggle_speech_recognition()
        else:
            # Normal metin
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(f"{text} ")
            self.text_edit.setTextCursor(cursor)
            self.text_edit.ensureCursorVisible()
    
    def clear_text(self):
        """Metin kutusunu temizler"""
        self.text_edit.clear()
        self.status_bar.showMessage("Metin temizlendi")
    
    def save_text(self):
        """Metni dosyaya kaydeder"""
        text = self.text_edit.toPlainText()
        if not text:
            self.status_bar.showMessage("Kaydedilecek metin yok")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Metni Kaydet", "", "Metin Dosyaları (*.txt);;Tüm Dosyalar (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(text)
                self.status_bar.showMessage(f"Metin kaydedildi: {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Kaydetme hatası: {str(e)}")
    
    def closeEvent(self, event):
        """Uygulama kapatılırken çağrılır"""
        # Modülleri temizle
        if self.eye_tracker and self.eye_tracking_active:
            self.eye_tracker.stop()
            
        if self.speech_recognizer and self.speech_recognition_active:
            self.speech_recognizer.stop()
            
        event.accept() 