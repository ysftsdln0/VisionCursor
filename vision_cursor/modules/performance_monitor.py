#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionCursor Performans İzleme Sistemi

Bu modül göz takibi ve ses tanıma performansını izler
"""

import time
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        self.fps_counter = deque(maxlen=30)  # Son 30 frame için FPS
        self.speech_accuracy = deque(maxlen=10)  # Son 10 tanıma için doğruluk
        self.cpu_usage = deque(maxlen=60)  # Son 60 saniye CPU kullanımı
        self.memory_usage = deque(maxlen=60)  # Son 60 saniye RAM kullanımı
        
        self.last_frame_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Performans izlemeyi başlat"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Performans izlemeyi durdur"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def record_frame(self):
        """Yeni frame kaydı"""
        current_time = time.time()
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_counter.append(fps)
        self.last_frame_time = current_time
    
    def record_speech_accuracy(self, expected_words, recognized_words):
        """Ses tanıma doğruluğunu kaydet"""
        if not expected_words or not recognized_words:
            return
            
        # Basit word accuracy hesaplaması
        expected_set = set(expected_words.lower().split())
        recognized_set = set(recognized_words.lower().split())
        
        if expected_set:
            accuracy = len(expected_set.intersection(recognized_set)) / len(expected_set)
            self.speech_accuracy.append(accuracy)
    
    def _monitor_system(self):
        """Sistem kaynaklarını izle"""
        while self.monitoring:
            try:
                # CPU kullanımı
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                
                # Bellek kullanımı
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
            except Exception as e:
                print(f"Sistem izleme hatası: {e}")
                time.sleep(1)
    
    def get_stats(self):
        """Performans istatistiklerini al"""
        stats = {
            'fps': {
                'current': self.fps_counter[-1] if self.fps_counter else 0,
                'average': sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0,
                'min': min(self.fps_counter) if self.fps_counter else 0,
                'max': max(self.fps_counter) if self.fps_counter else 0
            },
            'speech_accuracy': {
                'average': sum(self.speech_accuracy) / len(self.speech_accuracy) if self.speech_accuracy else 0,
                'min': min(self.speech_accuracy) if self.speech_accuracy else 0,
                'max': max(self.speech_accuracy) if self.speech_accuracy else 0
            },
            'cpu_usage': {
                'current': self.cpu_usage[-1] if self.cpu_usage else 0,
                'average': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            },
            'memory_usage': {
                'current': self.memory_usage[-1] if self.memory_usage else 0,
                'average': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            }
        }
        return stats
    
    def print_stats(self):
        """İstatistikleri yazdır"""
        stats = self.get_stats()
        
        print("\n=== VisionCursor Performans İstatistikleri ===")
        print(f"FPS: {stats['fps']['current']:.1f} (Ort: {stats['fps']['average']:.1f})")
        print(f"Ses Doğruluğu: %{stats['speech_accuracy']['average']*100:.1f}")
        print(f"CPU Kullanımı: %{stats['cpu_usage']['current']:.1f} (Ort: %{stats['cpu_usage']['average']:.1f})")
        print(f"RAM Kullanımı: %{stats['memory_usage']['current']:.1f} (Ort: %{stats['memory_usage']['average']:.1f})")
        print("=" * 45)
