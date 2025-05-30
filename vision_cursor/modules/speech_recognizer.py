import speech_recognition as sr
import whisper
import threading
import queue
import os
import tempfile
import time
import numpy as np
import logging

class SpeechRecognizer:
    def __init__(self, language="tr", use_whisper=True, callback=None):
        self.recognizer = sr.Recognizer()
        self.language = language  # Türkçe
        self.use_whisper = use_whisper
        self.callback = callback
        
        # Whisper modeli - Türkçe için optimize edilmiş ayarlar
        if use_whisper:
            try:
                # "base" model daha iyi doğruluk sağlar
                self.whisper_model = whisper.load_model("base")
                print("Whisper 'base' modeli yüklendi")
            except Exception as e:
                print(f"Whisper 'base' modeli yüklenemedi, 'tiny' modeli deneniyor: {e}")
                self.whisper_model = whisper.load_model("tiny")
        
        # Ses tanıma ayarları
        self.is_listening = False
        self.thread = None
        self.audio_queue = queue.Queue()
        self.temp_dir = tempfile.gettempdir()
        
        # Geliştirilmiş ses algılama parametreleri
        self.silence_threshold = 0.8  # Daha kısa bekleme
        self.energy_threshold = 400   # Daha hassas ses algılama
        self.pause_threshold = 0.6    # Kelimeler arası duraklama
        self.phrase_time_limit = 8    # Maksimum cümle süresi
        
        # Türkçe kelime filtreleme - daha kapsamlı liste
        self.turkish_filter_words = [
            'ne', 'nah', 'ah', 'eh', 'mm', 'hmm', 'uh', 'oh', 'hı', 'hım',
            'aaa', 'eee', 'iii', 'ooo', 'uuu', 'şşş', 'tss', 'pff'
        ]
        
        # Minimum kelime uzunluğu
        self.min_word_length = 2
        self.min_sentence_length = 3
        
        # Komut listesi
        self.commands = {
            "temizle": "clear",
            "sil": "clear", 
            "başlat": "start",
            "durdur": "stop",
            "bitir": "stop",
            "kaydet": "save",
            "göz takibini başlat": "start_eye",
            "göz takibini durdur": "stop_eye", 
            "ses tanımayı başlat": "start_speech",
            "ses tanımayı durdur": "stop_speech",
            "kapat": "quit",
            "çık": "quit"
        }
        
    def start(self):
        if self.is_listening:
            return
            
        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_and_recognize)
        self.thread.daemon = True
        self.thread.start()
        print("Ses tanıma başlatıldı")
        
    def stop(self):
        self.is_listening = False
        if self.thread:
            self.thread.join()
        print("Ses tanıma durduruldu")
    
    def _listen_and_recognize(self):
        """Geliştirilmiş ses dinleme ve tanıma fonksiyonu"""
        try:
            with sr.Microphone() as source:
                # Ortam gürültüsüne göre kalibre et
                print("Ortam gürültüsü kalibre ediliyor...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
                # Ses tanıma parametrelerini ayarla
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.energy_threshold = self.energy_threshold
                self.recognizer.pause_threshold = self.pause_threshold
                
                print("Dinleme başladı... (Konuşabilirsiniz)")
                
                while self.is_listening:
                    try:
                        # Konuşma dinle
                        audio = self.recognizer.listen(
                            source, 
                            timeout=5, 
                            phrase_time_limit=self.phrase_time_limit
                        )
                        
                        # Ses işleme thread'ini başlat
                        processing_thread = threading.Thread(
                            target=self._process_audio, 
                            args=(audio,)
                        )
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Dinleme hatası: {e}")
                        time.sleep(0.1)
                        continue
                        
        except Exception as e:
            print(f"Mikrofon başlatma hatası: {e}")
    
    def _process_audio(self, audio):
        """Ses verisini işle ve metne çevir"""
        try:
            text = ""
            
            if self.use_whisper:
                # Whisper ile tanıma
                text = self._whisper_recognize(audio)
            else:
                # Google Speech Recognition ile tanıma
                text = self._google_recognize(audio)
            
            if text:
                # Metni temizle ve filtrele
                cleaned_text = self._clean_text(text)
                
                # Türkçe tanıma iyileştirmeleri uygula
                improved_text = self.improve_turkish_recognition(cleaned_text)
                
                if improved_text and len(improved_text.strip()) > 1:
                    print(f"Tanınan metin: '{improved_text}'")
                    
                    # Komut kontrolü
                    if self._process_commands(improved_text):
                        return
                    
                    # Callback'i çağır
                    if self.callback:
                        self.callback(improved_text)
                        
        except Exception as e:
            print(f"Ses işleme hatası: {e}")
    
    def _whisper_recognize(self, audio):
        """Whisper ile ses tanıma"""
        try:
            # Geçici dosya oluştur
            temp_file = os.path.join(self.temp_dir, f"audio_{time.time()}.wav")
            
            with open(temp_file, "wb") as f:
                f.write(audio.get_wav_data())
            
            # Whisper ile tanıma - Türkçe optimize edilmiş parametreler
            result = self.whisper_model.transcribe(
                temp_file, 
                language=self.language,
                fp16=False,  # macOS uyumluluğu için
                temperature=0.0,  # Daha tutarlı sonuçlar için
                condition_on_previous_text=False,  # Her tanıma bağımsız
                no_speech_threshold=0.4,  # Sessizlik algılama
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            
            text = result["text"].strip()
            
            # Geçici dosyayı sil
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return text
            
        except Exception as e:
            print(f"Whisper tanıma hatası: {e}")
            return ""
    
    def _google_recognize(self, audio):
        """Google Speech Recognition ile ses tanıma"""
        try:
            text = self.recognizer.recognize_google(
                audio, 
                language=self.language,
                show_all=False
            )
            return text
            
        except sr.UnknownValueError:
            print("Konuşma anlaşılamadı")
            return ""
        except sr.RequestError as e:
            print(f"Google Speech Recognition hizmeti hatası: {e}")
            return ""
    
    def _clean_text(self, text):
        """Metni temizle ve filtrele"""
        if not text:
            return ""
            
        # Küçük harfe çevir
        text = text.lower().strip()
        
        # Gereksiz karakterleri temizle
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Çok uzun tekrarları kısalt
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)  # 10'dan fazla tekrar varsa 3'e düşür
        
        # Türkçe filtreleme kelimelerini kontrol et
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Çok kısa kelimeleri filtrele
            if len(word) < self.min_word_length:
                continue
                
            # Filtreleme listesindeki kelimeleri atla
            if word in self.turkish_filter_words:
                continue
                
            # Aynı karakterin çok tekrarını filtrele (örn: "hıhıhıhı...")
            if len(set(word)) == 1 and len(word) > 3:
                continue
                
            filtered_words.append(word)
        
        result = ' '.join(filtered_words)
        
        # Minimum cümle uzunluğu kontrolü
        if len(result) < self.min_sentence_length:
            return ""
            
        return result
    
    def _process_commands(self, text):
        """Komutları işle"""
        text_lower = text.lower()
        
        for command_text, command_action in self.commands.items():
            if command_text in text_lower:
                print(f"Komut algılandı: {command_text} -> {command_action}")
                
                if self.callback:
                    self.callback(text, is_command=True, command=command_action)
                return True
                
        return False
    
    def test_microphone(self):
        """Mikrofonu test et"""
        print("Mikrofon testi başlatılıyor...")
        try:
            with sr.Microphone() as source:
                print("Mikrofon bulundu:", source)
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Ortam gürültü seviyesi: {self.recognizer.energy_threshold}")
                
                print("3 saniye konuşun...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                print("Ses kaydedildi, işleniyor...")
                
                # Test tanıma
                try:
                    text = self.recognizer.recognize_google(audio, language="tr-TR")
                    print(f"Google tanıma sonucu: '{text}'")
                except:
                    print("Google tanıma başarısız")
                
                if self.use_whisper:
                    try:
                        temp_file = os.path.join(self.temp_dir, f"test_audio_{time.time()}.wav")
                        with open(temp_file, "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        result = self.whisper_model.transcribe(temp_file, language="tr")
                        print(f"Whisper tanıma sonucu: '{result['text']}'")
                        
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        print(f"Whisper tanıma hatası: {e}")
                        
        except Exception as e:
            print(f"Mikrofon test hatası: {e}")
    
    def get_available_microphones(self):
        """Mevcut mikrofonları listele"""
        print("Mevcut ses cihazları:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  {index}: {name}")
        return sr.Microphone.list_microphone_names()
    
    def improve_turkish_recognition(self, text):
        """Türkçe tanıma sonuçlarını iyileştir"""
        if not text:
            return text
            
        # Türkçe karakter düzeltmeleri
        corrections = {
            'i': 'ı', 'I': 'İ',
            'g': 'ğ', 'G': 'Ğ',
            'u': 'ü', 'U': 'Ü',
            'o': 'ö', 'O': 'Ö',
            's': 'ş', 'S': 'Ş',
            'c': 'ç', 'C': 'Ç'
        }
        
        # Yaygın Türkçe kelime düzeltmeleri
        word_corrections = {
            'ben': 'ben',
            'sen': 'sen',
            'o': 'o',
            'biz': 'biz',
            'siz': 'siz',
            'onlar': 'onlar',
            've': 've',
            'ile': 'ile',
            'bir': 'bir',
            'bu': 'bu',
            'şu': 'şu',
            'merhaba': 'merhaba',
            'selam': 'selam',
            'tamam': 'tamam',
            'evet': 'evet',
            'hayır': 'hayır',
            'başlat': 'başlat',
            'durdur': 'durdur',
            'temizle': 'temizle'
        }
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Kelime düzeltmelerini uygula
            corrected_word = word_corrections.get(word.lower(), word)
            corrected_words.append(corrected_word)
        
        return ' '.join(corrected_words)
    
    def set_microphone_by_index(self, mic_index):
        """Belirli bir mikrofonu seç"""
        try:
            self.microphone = sr.Microphone(device_index=mic_index)
            print(f"Mikrofon {mic_index} seçildi")
            return True
        except Exception as e:
            print(f"Mikrofon {mic_index} seçilemedi: {e}")
            return False