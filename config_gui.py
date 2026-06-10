import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import speech_recognition as sr
import sounddevice as sd
import numpy as np

CONFIG_FILE = "config.json"

def run_startup_dialog():
    config = {
        "camera_index": 0,
        "mic_index": 1,
        "always_on_top": True
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                config.update(saved)
        except Exception:
            pass
            
    mics = sr.Microphone.list_microphone_names()
    stream = None

    def save_and_close():
        nonlocal stream
        try:
            cam_idx = int(cam_var.get())
            mic_name = mic_var.get()
            if mic_name in mics:
                mic_idx = mics.index(mic_name)
            else:
                mic_idx = 1
                
            config["camera_index"] = cam_idx
            config["mic_index"] = mic_idx
            config["always_on_top"] = on_top_var.get()
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
                
            if stream:
                stream.stop()
                stream.close()
                
            root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Camera index must be a number.")

    def get_volume(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        if root.winfo_exists():
            root.after(0, update_meter, min(100, int(volume_norm)))

    def update_meter(vol):
        meter['value'] = vol

    def on_mic_change(event=None):
        nonlocal stream
        if stream:
            stream.stop()
            stream.close()
            stream = None
            
        mic_name = mic_var.get()
        if mic_name in mics:
            mic_idx = mics.index(mic_name)
            try:
                stream = sd.InputStream(callback=get_volume, device=mic_idx, channels=1)
                stream.start()
            except Exception as e:
                print("Could not start audio stream for meter:", e)
                meter['value'] = 0

    root = tk.Tk()
    root.title("Gesture Control Settings")
    root.geometry("400x250")
    root.eval('tk::PlaceWindow . center')
    
    # Camera
    ttk.Label(root, text="Camera Index:").pack(pady=(10, 0))
    cam_var = tk.StringVar(value=str(config["camera_index"]))
    cam_entry = ttk.Spinbox(root, from_=0, to=10, textvariable=cam_var, width=10)
    cam_entry.pack(pady=5)
    
    # Mic
    ttk.Label(root, text="Microphone:").pack(pady=(10, 0))
    mic_var = tk.StringVar()
    if config["mic_index"] < len(mics):
        mic_var.set(mics[config["mic_index"]])
    elif mics:
        mic_var.set(mics[0])
        
    mic_combo = ttk.Combobox(root, textvariable=mic_var, values=mics, state="readonly", width=40)
    mic_combo.pack(pady=5)
    mic_combo.bind("<<ComboboxSelected>>", on_mic_change)
    
    # Audio Meter
    meter = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=250, maximum=100)
    meter.pack(pady=5)
    
    # Always on Top
    on_top_var = tk.BooleanVar(value=config["always_on_top"])
    ttk.Checkbutton(root, text="Always on Top", variable=on_top_var).pack(pady=5)
    
    ttk.Button(root, text="Start Application", command=save_and_close).pack(pady=10)
    
    # Start meter for default mic
    on_mic_change()
    
    # Handle window close (X button)
    def on_closing():
        nonlocal stream
        if stream:
            stream.stop()
            stream.close()
        root.destroy()
        import sys
        sys.exit(0) # Terminate if user closes window
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()
    
    return config
