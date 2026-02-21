import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, AutoModelForAudioClassification
from huggingface_hub import hf_hub_download
import importlib.util
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 16000


class StressDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"🔧 Using device: {self.device}")

        # Load Wav2Vec2 base via torchaudio (as specified by the model repo)
        print("📥 Loading Wav2Vec2 model (torchaudio)...")
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.w2v_model = bundle.get_model().to(self.device)
        self.w2v_model.eval()
        self.w2v_sample_rate = bundle.sample_rate
        print("✅ Wav2Vec2 loaded")

        # Load stress classification model
        print("📥 Loading stress classification model...")
        self.stress_model = self._load_stress_model()
        print("✅ Stress model loaded")

    def _load_stress_model(self):
        """Load the custom stress detection model from HuggingFace"""
        repo = "forwarder1121/voice-based-stress-recognition"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📥 Attempting to load stress model (attempt {attempt + 1}/{max_retries})...")

                # Download and load custom models.py so AutoConfig/AutoModel can find the classes
                code_path = hf_hub_download(repo_id=repo, filename="models.py")
                spec = importlib.util.spec_from_file_location("models", code_path)
                models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(models)

                # Load config + model with trust_remote_code
                config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
                model = AutoModelForAudioClassification.from_pretrained(
                    repo,
                    config=config,
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

    def load_audio(self, audio_path):
        """Load audio file and resample to 16kHz mono"""
        print(f"🎵 Loading audio from: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.w2v_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.w2v_sample_rate)

        duration = waveform.shape[1] / self.w2v_sample_rate
        print(f"✅ Audio loaded: {duration:.2f}s @ {self.w2v_sample_rate}Hz")
        return waveform

    def extract_w2v_embedding(self, waveform):
        """Extract 512-dim Wav2Vec2 CNN feature embeddings"""
        print("🧠 Extracting Wav2Vec2 embeddings...")

        waveform = waveform.to(self.device)

        with torch.no_grad():
            # extract_features returns a list of outputs from each transformer layer
            # but the feature_extractor (CNN) output is 512-dim, which is what the
            # stress model expects. Access it directly.
            features, _ = self.w2v_model.feature_extractor(waveform, length=None)
            # features shape: (batch, time, 512)
            # Mean pool over time -> (batch, 512)
            embedding = features.mean(dim=1)

        print(f"✅ Embedding shape: {embedding.shape}")
        return embedding

    def predict_stress(self, w2v_embedding):
        """Predict stress from 512-dim Wav2Vec2 embedding"""
        print("🔮 Running stress prediction...")

        with torch.no_grad():
            outputs = self.stress_model(w2v_embedding)
            probs = F.softmax(outputs.logits, dim=-1)

        return probs

    def analyze_audio(self, audio_path):
        """Complete pipeline: audio -> embeddings -> stress prediction"""
        try:
            # Load audio
            waveform = self.load_audio(audio_path)

            # Extract 512-dim W2V embeddings
            embedding = self.extract_w2v_embedding(waveform)

            # Predict stress
            probs = self.predict_stress(embedding)

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
