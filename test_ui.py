import os
import json
import re
import requests
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk


# CONFIG 

class ClinicalAppLogic:
    MODEL = "openrouter/auto"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    SYSTEM_PROMPT = """You are a medical information extraction assistant. Rules: Extract what is explicitly mentioned. RETURN STRICT JSON ONLY."""

    @staticmethod
    def call_llm(api_key, note_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Clinical-Extraction-Demo"
        }
        payload = {
            "model": ClinicalAppLogic.MODEL,
            "messages": [
                {"role": "system", "content": ClinicalAppLogic.SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured medical data from: {note_text}"}
            ],
            "temperature": 0
        }
        response = requests.post(ClinicalAppLogic.API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        content = re.sub(r"```json|```", "", content).strip()
        return json.loads(content)


# UI CLASS

class ClinicalParserUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Note Parser (OpenRouter)")
        self.root.geometry("800x700")
        self.root.configure(bg="#f3f4f6")

        # Variables
        self.file_path = tk.StringVar(value="No file selected")
        self.api_key = tk.StringVar(value=os.getenv("OPENROUTER_API_KEY", ""))

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1e293b", height=80)
        header.pack(fill=tk.X)
        tk.Label(header, text="Clinical Data Extractor", font=("Arial", 18, "bold"), fg="white", bg="#1e293b").pack(pady=20)

        # Input Frame
        input_frame = tk.LabelFrame(self.root, text=" Configuration & Input ", padx=20, pady=10, bg="#f3f4f6")
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(input_frame, text="API Key:", bg="#f3f4f6").grid(row=0, column=0, sticky="w")
        tk.Entry(input_frame, textvariable=self.api_key, show="*", width=50).grid(row=0, column=1, padx=10, pady=5)

        tk.Label(input_frame, text="Note File:", bg="#f3f4f6").grid(row=1, column=0, sticky="w")
        tk.Label(input_frame, textvariable=self.file_path, fg="#4b5563", wraplength=400).grid(row=1, column=1, sticky="w", padx=10)
        tk.Button(input_frame, text="Browse", command=self.browse_file).grid(row=1, column=2)

        # Action Button
        self.run_btn = tk.Button(self.root, text="Start Extraction", command=self.start_thread, 
                                 bg="#10b981", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10)
        self.run_btn.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=5)

        # Output Area
        tk.Label(self.root, text="Results (JSON):", bg="#f3f4f6", font=("Arial", 10, "bold")).pack(anchor="w", padx=25)
        self.output_area = scrolledtext.ScrolledText(self.root, height=20, bg="white", font=("Courier", 10))
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filename:
            self.file_path.set(filename)

    def start_thread(self):
        if not self.api_key.get():
            messagebox.showerror("Error", "Please provide an API Key.")
            return
        if "No file selected" in self.file_path.get():
            messagebox.showerror("Error", "Please select a .txt file.")
            return
            
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, "Processing... please wait.\n")
        
        threading.Thread(target=self.process_cases, daemon=True).start()

    def process_cases(self):
        try:
            with open(self.file_path.get(), "r") as f:
                text = f.read()
            
            # Use regex to split cases as per your original logic
            cases = [c.strip() for c in re.split(r"(?i)CASE \d+:?", text) if c.strip()]
            results = []

            for idx, case in enumerate(cases, start=1):
                try:
                    data = ClinicalAppLogic.call_llm(self.api_key.get(), case)
                    results.append({"case_id": idx, "data": data})
                except Exception as e:
                    results.append({"case_id": idx, "error": str(e)})

            final_json = json.dumps(results, indent=2)
            
            # Update UI from thread safely
            self.root.after(0, lambda: self.update_results(final_json))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("File Error", str(e)))
        finally:
            self.root.after(0, self.stop_loading)

    def update_results(self, content):
        self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, content)
        
    def stop_loading(self):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClinicalParserUI(root)
    root.mainloop()