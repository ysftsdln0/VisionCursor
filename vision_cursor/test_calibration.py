#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionCursor Test ve Kalibrasyon Aracı

Bu script, göz takibi ve ses tanıma sistemlerini test etmek 
ve kalibre etmek için kullanılır.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.eye_tracker import EyeTracker
from modules.speech_recognizer import SpeechRecognizer
from modules.camera import Camera
import cv2

def test_camera():
    """Kamera testini gerçekleştirir"""
    print("=== KAMERA TESTİ ===")
    camera = Camera()
    
    if camera.start():
        print("✓ Kamera başarıyla başlatıldı")
        
        # 5 saniye boyunca frame al
        start_time = time.time()
        frame_count = 0
        
        def frame_callback(frame):
            nonlocal frame_count
            frame_count += 1
            
        camera.start(callback=frame_callback)
        time.sleep(5)
        camera.stop()
        
        fps = frame_count / 5
        print(f"✓ FPS: {fps:.1f}")
        print("✓ Kamera testi başarılı")
        return True
    else:
        print("✗ Kamera başlatılamadı")
        return False

def test_eye_tracking():
    """Göz takibi testini gerçekleştirir"""
    print("\n=== GÖZ TAKİBİ TESTİ ===")
    
    try:
        eye_tracker = EyeTracker()
        eye_tracker.start()
        print("✓ Göz takibi başlatıldı")
        
        print("10 saniye boyunca göz takibi test ediliyor...")
        print("Gözlerinizi hareket ettirin, 2 saniye sabit bakın...")
        
        time.sleep(10)
        
        eye_tracker.stop()
        print("✓ Göz takibi testi tamamlandı")
        return True
        
    except Exception as e:
        print(f"✗ Göz takibi hatası: {e}")
        return False

def test_speech_recognition():
    """Ses tanıma testini gerçekleştirir"""
    print("\n=== SES TANIMA TESTİ ===")
    
    recognized_texts = []
    
    def speech_callback(text, is_command=False, command=None):
        if is_command:
            print(f"Komut algılandı: {command}")
        else:
            print(f"Tanınan metin: '{text}'")
            recognized_texts.append(text)
    
    try:
        speech_recognizer = SpeechRecognizer(callback=speech_callback)
        speech_recognizer.start()
        print("✓ Ses tanıma başlatıldı")
        
        print("10 saniye boyunca konuşun...")
        print("Test kelimeleri: 'merhaba', 'test', 'temizle', 'kaydet'")
        
        time.sleep(10)
        
        speech_recognizer.stop()
        print("✓ Ses tanıma testi tamamlandı")
        print(f"Toplam {len(recognized_texts)} metin tanındı")
        return True
        
    except Exception as e:
        print(f"✗ Ses tanıma hatası: {e}")
        return False

def calibration_mode():
    """Kalibrasyon modu"""
    print("\n=== KALİBRASYON MODU ===")
    
    try:
        eye_tracker = EyeTracker()
        eye_tracker.start()
        
        print("Kalibrasyon başlatıldı...")
        print("Ekranın köşelerine bakın ve pozisyonları kontrol edin")
        print("Ayarları yapmak için aşağıdaki tuşları kullanın:")
        print("W/S: Y offset ayarlama")
        print("A/D: X offset ayarlama") 
        print("Q/E: Y scale ayarlama")
        print("Z/C: X scale ayarlama")
        print("R: Sıfırla")
        print("ESC: Çık")
        
        offset_x, offset_y = 0, 0
        scale_x, scale_y = 1.0, 1.0
        
        while True:
            # Gerçek uygulamada klavye inputu için farklı bir library gerekir
            # Burada sadece placeholder
            print(f"Mevcut ayarlar: offset({offset_x}, {offset_y}), scale({scale_x:.2f}, {scale_y:.2f})")
            
            user_input = input("Ayar girin (w/s/a/d/q/e/z/c/r/x): ").lower()
            
            if user_input == 'w':
                offset_y -= 10
            elif user_input == 's':
                offset_y += 10
            elif user_input == 'a':
                offset_x -= 10
            elif user_input == 'd':
                offset_x += 10
            elif user_input == 'q':
                scale_y -= 0.1
            elif user_input == 'e':
                scale_y += 0.1
            elif user_input == 'z':
                scale_x -= 0.1
            elif user_input == 'c':
                scale_x += 0.1
            elif user_input == 'r':
                offset_x, offset_y = 0, 0
                scale_x, scale_y = 1.0, 1.0
            elif user_input == 'x':
                break
            
            eye_tracker.calibrate(offset_x, offset_y, scale_x, scale_y)
        
        eye_tracker.stop()
        print("Kalibrasyon tamamlandı")
        return True
        
    except Exception as e:
        print(f"✗ Kalibrasyon hatası: {e}")
        return False

def main():
    print("VisionCursor Test ve Kalibrasyon Aracı")
    print("=====================================")
    
    while True:
        print("\nMenü:")
        print("1. Kamera testi")
        print("2. Göz takibi testi") 
        print("3. Ses tanıma testi")
        print("4. Kalibrasyon modu")
        print("5. Tam sistem testi")
        print("0. Çıkış")
        
        choice = input("\nSeçiminiz (0-5): ").strip()
        
        if choice == "0":
            print("Çıkış yapılıyor...")
            break
        elif choice == "1":
            test_camera()
        elif choice == "2":
            test_eye_tracking()
        elif choice == "3":
            test_speech_recognition()
        elif choice == "4":
            calibration_mode()
        elif choice == "5":
            print("\n=== TAM SİSTEM TESTİ ===")
            camera_ok = test_camera()
            eye_ok = test_eye_tracking() if camera_ok else False
            speech_ok = test_speech_recognition()
            
            print(f"\n=== SONUÇ ===")
            print(f"Kamera: {'✓' if camera_ok else '✗'}")
            print(f"Göz takibi: {'✓' if eye_ok else '✗'}")
            print(f"Ses tanıma: {'✓' if speech_ok else '✗'}")
            
            if camera_ok and eye_ok and speech_ok:
                print("🎉 Tüm testler başarılı!")
            else:
                print("⚠️  Bazı testler başarısız")
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main()
