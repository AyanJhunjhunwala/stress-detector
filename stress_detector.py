import torch
import torch.nn.functional as F
import librosa
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2Processor,
    AutoConfig, 
    AutoModelForAudioClassification
)
from huggingface_hub import hf_hub_download
import importlib.util
import warnings
warnings.filterwarnings('ignore')


class StressDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"🔧 Using device: {self.device}")
        
        # Load Wave2Vec2 for feature extraction
        print("📥 Loading Wave2Vec2 model...")
        self.w2v_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.w2v_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)
        self.w2v_model.eval()
        print("✅ Wave2Vec2 loaded")
        
        # Load stress classification model
        print("📥 Loading stress classification model...")
        self.stress_model = self._load_stress_model()
        print("✅ Stress model loaded")
        
        # Create projection layer for dimension matching
        self.projection = torch.nn.Linear(768, 512).to(self.device)
        
    def _load_stress_model(self):
        """Load the custom stress detection model from HuggingFace"""
        repo = "forwarder1121/voice-based-stress-recognition"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📥 Attempting to load stress model (attempt {attempt + 1}/{max_retries})...")
                
                # Download and load custom models.py
                code_path = hf_hub_download(repo_id=repo, filename="models.py")
                spec = importlib.util.spec_from_file_location("models", code_path)
                models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(models)
                
                # Load model
                model = AutoModelForAudioClassification.from_pretrained(
                    repo,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).to(self.device)
                model.eval()
                
                print("✅ Stress model loaded successfully")
                return model
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)
                else:
                    raise
    
    def load_audio(self, audio_path, target_sr=16000):
        """Load and preprocess audio file"""
        print(f"🎵 Loading audio from: {audio_path}")
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            print(f"✅ Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")
            return audio, sr
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            raise
    
    def extract_w2v_embedding(self, audio, sr=16000):
        """Extract Wave2Vec2 embeddings from audio"""
        print("🧠 Extracting Wave2Vec2 embeddings...")
        
        try:
            # Process audio
            inputs = self.w2v_processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.w2v_model(input_values)
                # Mean pool over time dimension: (batch, time, 768) -> (batch, 768)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            print(f"✅ Embedding shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error extracting embeddings: {e}")
            raise
    
    def predict_stress(self, w2v_embedding):
        """Predict stress from Wave2Vec2 embedding"""
        print("🔮 Running stress prediction...")
        
        try:
            # Project from 768 to 512 dimensions
            if w2v_embedding.shape[1] == 768:
                w2v_embedding = self.projection(w2v_embedding)
                print(f"✅ Projected to shape: {w2v_embedding.shape}")
            
            # Run inference
            with torch.no_grad():
                outputs = self.stress_model(w2v_embedding)
                probs = F.softmax(outputs.logits, dim=-1)
            
            return probs
            
        except Exception as e:
            print(f"❌ Error predicting stress: {e}")
            raise
    
    def analyze_audio(self, audio_path):
        """Complete pipeline: audio -> embeddings -> stress prediction"""
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            # Extract W2V embeddings
            embeddings = self.extract_w2v_embedding(audio, sr)
            
            # Predict stress
            probs = self.predict_stress(embeddings)
            
            # Calculate results
            not_stressed_prob = probs[0, 0].item() * 100
            stressed_prob = probs[0, 1].item() * 100
            
            result = "STRESSED" if stressed_prob > not_stressed_prob else "NOT STRESSED"
            confidence = max(stressed_prob, not_stressed_prob)
            
            # Display results
            print("\n" + "="*60)
            print("📊 STRESS DETECTION RESULTS")
            print("="*60)
            print(f"😌 Not Stressed: {not_stressed_prob:.2f}%")
            print(f"😰 Stressed:     {stressed_prob:.2f}%")
            print("="*60)
            print(f"🎯 Prediction: {result} (confidence: {confidence:.2f}%)")
            print("="*60 + "\n")
            
            return {
                'not_stressed': not_stressed_prob,
                'stressed': stressed_prob,
                'prediction': result,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"❌ Error in analysis pipeline: {e}")
            raise