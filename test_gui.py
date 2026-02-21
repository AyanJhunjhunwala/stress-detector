import tkinter as tk
import threading
import wave
import tempfile
import os
import subprocess
import base64
import json
import signal

SAMPLE_RATE = 16000
CHANNELS = 1


class StressTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress Detector Tester")
        self.root.geometry("420x520")
        self.root.resizable(False, False)

        self.recording = False
        self.rec_process = None
        self.tmp_wav = None

        # --- Mode toggle ---
        mode_frame = tk.LabelFrame(root, text="Inference Mode", padx=10, pady=5)
        mode_frame.pack(fill="x", padx=15, pady=(15, 5))

        self.mode = tk.StringVar(value="local")
        tk.Radiobutton(mode_frame, text="Local (direct Python)", variable=self.mode, value="local").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Docker (RunPod container)", variable=self.mode, value="docker").pack(anchor="w")

        # --- Record controls ---
        rec_frame = tk.LabelFrame(root, text="Recording", padx=10, pady=10)
        rec_frame.pack(fill="x", padx=15, pady=10)

        self.rec_btn = tk.Button(rec_frame, text="Hold to Record", font=("Helvetica", 14, "bold"),
                                 bg="#e74c3c", fg="white", activebackground="#c0392b",
                                 width=20, height=2)
        self.rec_btn.pack(pady=5)
        self.rec_btn.bind("<ButtonPress-1>", self.start_recording)
        self.rec_btn.bind("<ButtonRelease-1>", self.stop_recording)

        self.rec_status = tk.Label(rec_frame, text="Ready", font=("Helvetica", 11))
        self.rec_status.pack()

        # --- Analyze button ---
        self.analyze_btn = tk.Button(root, text="Analyze", font=("Helvetica", 13, "bold"),
                                     bg="#2ecc71", fg="white", activebackground="#27ae60",
                                     width=20, height=2, state="disabled",
                                     command=self.run_analysis)
        self.analyze_btn.pack(pady=10)

        # --- Results ---
        res_frame = tk.LabelFrame(root, text="Results", padx=10, pady=10)
        res_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.result_text = tk.Text(res_frame, height=10, font=("Courier", 11), state="disabled", wrap="word")
        self.result_text.pack(fill="both", expand=True)

    # ---- Recording via ffmpeg (no pip dependencies) ----

    def start_recording(self, event=None):
        self.tmp_wav = tempfile.mktemp(suffix=".wav")
        self.rec_status.config(text="Recording...", fg="red")
        self.analyze_btn.config(state="disabled")

        # Use ffmpeg to record from default macOS mic input
        self.rec_process = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "avfoundation",
                "-i", ":default",
                "-ac", str(CHANNELS),
                "-ar", str(SAMPLE_RATE),
                "-sample_fmt", "s16",
                self.tmp_wav,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.recording = True

    def stop_recording(self, event=None):
        if not self.recording or self.rec_process is None:
            return
        self.recording = False

        # Send 'q' to ffmpeg to stop gracefully
        try:
            self.rec_process.stdin.write(b"q")
            self.rec_process.stdin.flush()
        except BrokenPipeError:
            pass
        self.rec_process.wait(timeout=5)
        self.rec_process = None

        if not os.path.exists(self.tmp_wav) or os.path.getsize(self.tmp_wav) < 100:
            self.rec_status.config(text="No audio captured", fg="orange")
            return

        # Read duration from wav
        try:
            with wave.open(self.tmp_wav, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
            self.rec_status.config(text=f"Recorded {duration:.1f}s", fg="green")
        except Exception:
            self.rec_status.config(text="Recorded (unknown length)", fg="green")

        self.analyze_btn.config(state="normal")

    # ---- Analysis ----

    def run_analysis(self):
        self.analyze_btn.config(state="disabled")
        self._set_result("Running inference...")
        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        try:
            if self.mode.get() == "local":
                result = self._run_local()
            else:
                result = self._run_docker()
            self.root.after(0, self._show_result, result)
        except Exception as e:
            self.root.after(0, self._set_result, f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.analyze_btn.config(state="normal"))

    def _run_local(self):
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from stress_detector import StressDetector

        if not hasattr(self, "_detector"):
            self._set_result("Loading models (first run)...")
            self._detector = StressDetector()

        return self._detector.analyze_audio(self.tmp_wav)

    def _run_docker(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build image if needed
        self._set_result("Building Docker image...")
        subprocess.run(
            ["docker", "build", "-t", "stress-detector", "."],
            cwd=script_dir, check=True, capture_output=True
        )

        # Encode audio as base64
        with open(self.tmp_wav, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = json.dumps({"input": {"audio_base64": audio_b64}})

        self._set_result("Running inference in Docker...")

        # Run container, invoke handler directly instead of RunPod server
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--gpus", "all",
                "-e", f"PAYLOAD={payload}",
                "stress-detector",
                "python3", "-u", "-c",
                (
                    "import json, os; "
                    "payload = json.loads(os.environ['PAYLOAD']); "
                    "from handler import handler; "
                    "result = handler(payload); "
                    "print('RESULT_JSON:' + json.dumps(result))"
                ),
            ],
            capture_output=True, text=True, timeout=300
        )

        # If GPU not available, retry without --gpus
        if result.returncode != 0 and "could not select device driver" in (result.stderr or ""):
            self._set_result("No GPU detected, retrying on CPU...")
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-e", f"PAYLOAD={payload}",
                    "stress-detector",
                    "python3", "-u", "-c",
                    (
                        "import json, os; "
                        "payload = json.loads(os.environ['PAYLOAD']); "
                        "from handler import handler; "
                        "result = handler(payload); "
                        "print('RESULT_JSON:' + json.dumps(result))"
                    ),
                ],
                capture_output=True, text=True, timeout=300
            )

        if result.returncode != 0:
            raise RuntimeError(f"Docker failed:\n{result.stderr[-500:]}")

        for line in result.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])

        raise RuntimeError(f"No result found in Docker output:\n{result.stdout[-500:]}")

    # ---- Display ----

    def _show_result(self, result):
        if isinstance(result, dict) and "results" in result:
            r = result["results"]
        elif isinstance(result, dict) and "prediction" in result:
            r = result
        else:
            self._set_result(json.dumps(result, indent=2))
            return

        lines = [
            f"  Not Stressed:  {r['not_stressed']:.1f}%",
            f"  Stressed:      {r['stressed']:.1f}%",
            "",
            f"  Prediction:    {r['prediction']}",
            f"  Confidence:    {r['confidence']:.1f}%",
        ]
        self._set_result("\n".join(lines))

    def _set_result(self, text):
        def _update():
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", text)
            self.result_text.config(state="disabled")
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)


if __name__ == "__main__":
    root = tk.Tk()
    app = StressTestGUI(root)
    root.mainloop()
