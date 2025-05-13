import speech_recognition as sr
import whisper
import threading
import queue
import os
import tempfile
import time
import numpy as np

class SpeechRecognizer:
    def __init__(self, language="tr", use_whisper=True, callback=None):
        self.recognizer = sr.Recognizer()
        self.language = language  # Türkçe
        self.use_whisper = use_whisper
        self.callback = callback  # Tanınan metnin gönderileceği callback
        
        # Whisper modeli
        if use_whisper:
            # "tiny" model en hızlı çalışan modeldir, daha doğru sonuçlar için "base" veya "small" kullanılabilir
            self.whisper_model = whisper.load_model("tiny")
        
        # İşlem için gerekli değişkenler
        self.is_listening = False
        self.thread = None
        self.audio_queue = queue.Queue()
        self.temp_dir = tempfile.gettempdir()
        
        # Kullanıcının konuşmayı bitirdiğini algılamak için değişkenler
        self.silence_threshold = 1.0  # saniye
        self.energy_threshold = 300  # Ses seviyesi eşiği
        
    def start(self):
        if self.is_listening:
            return
            
        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_and_recognize)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.is_listening = False
        if self.thread:
            self.thread.join()
    
    def _listen_and_recognize(self):
        with sr.Microphone() as source:
            # Ortam gürültüsüne göre ayarlama
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = self.energy_threshold
            
            print("Dinleme başladı...")
            
            while self.is_listening:
                try:
                    # Konuşma algılanması için dinleme başlat
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Ses verisini işleme kuyruğuna ekle
                    self.audio_queue.put(audio)
                    
                    # Ses işleme thread'ini başlat
                    processing_thread = threading.Thread(target=self._process_audio)
                    processing_thread.daemon = True
                    processing_thread.start()
                    
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Dinleme hatası: {e}")
                    continue
    
    def _process_audio(self):
        if self.audio_queue.empty():
            return
            
        audio = self.audio_queue.get()
        
        try:
            if self.use_whisper:
                # Whisper için ses dosyasını geçici kaydet
                temp_file = os.path.join(self.temp_dir, f"audio_{time.time()}.wav")
                with open(temp_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Whisper ile tanıma
                result = self.whisper_model.transcribe(temp_file, language=self.language)
                text = result["text"].strip()
                
                # Geçici dosyayı sil
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                # Google Speech Recognition kullanarak tanıma
                text = self.recognizer.recognize_google(audio, language=self.language)
            
            # Boş değilse ve callback tanımlıysa çağır
            if text and self.callback:
                # Cümle sonuna nokta ekleme
                if not any(text.endswith(p) for p in ['.', '!', '?']):
                    text += '.'
                self.callback(text)
                
        except sr.UnknownValueError:
            print("Konuşma anlaşılamadı")
        except sr.RequestError as e:
            print(f"Google Speech Recognition hizmeti hatası: {e}")
        except Exception as e:
            print(f"Ses tanıma hatası: {e}")
    
    def process_commands(self, text):
        """Temel komutları işler ve True/False döndürür (komut bulundu/bulunmadı)"""
        # Küçük harfe çevir
        text = text.lower()
        
        # Komut listesi
        commands = {
            "temizle": "clear",
            "sil": "clear",
            "başlat": "start",
            "durdur": "stop",
            "bitir": "stop",
            "kaydet": "save",
            "göz takibini başlat": "start_eye",
            "göz takibini durdur": "stop_eye",
            "ses tanımayı başlat": "start_speech",
            "ses tanımayı durdur": "stop_speech"
        }
        
        # Komut kontrolü
        for command_text, command_action in commands.items():
            if command_text in text:
                if self.callback:
                    self.callback(text, is_command=True, command=command_action)
                return True
                
        return False 