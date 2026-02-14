import runpod
import torch
import base64
import tempfile
import os
import traceback
from stress_detector import StressDetector

# Initialize detector once (stays loaded between requests)
print("="*60)
print("🚀 Initializing Stress Detector...")
print("="*60)

detector = None

def get_detector():
    """Get or initialize the stress detector, retrying if previous init failed."""
    global detector
    if detector is None:
        detector = StressDetector()
        print("="*60)
        print("✅ Stress Detector Ready!")
        print("="*60)
    return detector

try:
    get_detector()
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    traceback.print_exc()


def handler(event):
    """
    RunPod serverless handler function
    
    Expected input:
    {
        "input": {
            "audio_base64": "<base64 encoded audio file>"
        }
    }
    
    Returns:
    {
        "status": "success",
        "results": {
            "not_stressed": 45.2,
            "stressed": 54.8,
            "prediction": "STRESSED",
            "confidence": 54.8
        }
    }
    """
    print("\n" + "="*60)
    print("📨 Received new request")
    print("="*60)
    
    try:
        active_detector = get_detector()
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": f"Detector not initialized: {e}"
        }

    try:
        # Get base64 audio from input
        audio_base64 = event["input"].get("audio_base64")
        
        if not audio_base64:
            return {
                "status": "error",
                "error": "No audio_base64 provided in input"
            }
        
        print(f"📦 Received audio data: {len(audio_base64)} characters")
        
        # Decode base64 to audio bytes
        audio_bytes = base64.b64decode(audio_base64)
        print(f"✅ Decoded {len(audio_bytes)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        print(f"💾 Saved to temporary file: {tmp_path}")
        
        try:
            # Run stress detection
            results = active_detector.analyze_audio(tmp_path)
            
            print("✅ Analysis complete!")
            
            return {
                "status": "success",
                "results": results
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🗑️  Cleaned up temp file")
    
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 Starting RunPod Serverless Handler")
    print("="*60 + "\n")
    runpod.serverless.start({"handler": handler})