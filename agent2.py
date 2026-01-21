"""
================================================================================
AUTONOMOUS PATIENT RISK REVIEW & RECOMMEND AGENT SYSTEM
================================================================================
Complete Single-File Implementation

Integrates:
- Problem 1: ML-based risk prediction (XGBoost)
- Problem 2: LLM-based clinical extraction (OpenRouter API)
- Multi-agent orchestration for clinical decision support

INSTALLATION:
    pip install pandas numpy scikit-learn xgboost requests

SETUP:
    1. (Optional) Set API key: export OPENROUTER_API_KEY=your_key
    2. (Optional) Place trained model: xgboost_model.pkl
    3. Run: python agent_system.py

The system auto-generates required config and data files on first run.

================================================================================
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import json
import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Optional imports with fallbacks
try:
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠ XGBoost not installed - using simulated predictions")

try:
    import requests
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("⚠ requests not installed - using rule-based extraction")


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """Manages all configuration and data files"""
    
    def __init__(self):
        self.config = self.load_config()
        self.guidelines = self.load_guidelines()
        self.patients = self.load_patients()
        self.ml_model = self.load_ml_model()
    
    def load_config(self):
        """Load or create config.json"""
        config_file = 'config.json'
        default = {
            "ml_model": {
                "model_path": "xgboost_model.pkl",
                "risk_thresholds": {"high": 70, "medium": 40, "low": 0},
                "weights": {"lab_result": 0.6, "procedure_priority": 8.0}
            },
            "llm_extractor": {
                "api_url": "https://openrouter.ai/api/v1/chat/completions",
                "model": "openrouter/auto",
                "temperature": 0,
                "api_key_env": "OPENROUTER_API_KEY"
            },
            "agent_delays": {
                "risk_scoring": 0.8,
                "clinical_understanding": 1.0,
                "guideline_reasoning": 0.6,
                "decision_making": 0.7
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            with open(config_file, 'w') as f:
                json.dump(default, f, indent=4)
            print(f"✓ Created: {config_file}")
            return default
    
    def load_guidelines(self):
        """Load or create clinical_guidelines.json"""
        file = 'clinical_guidelines.json'
        default = {
            "HIGH_RISK_CONGESTIVE_HEART_FAILURE": {
                "condition": "Congestive Heart Failure",
                "actions": [
                    "Schedule cardiology follow-up within 7 days",
                    "Daily weight monitoring",
                    "Review diuretic dosing",
                    "Assess volume status"
                ],
                "urgency": "high"
            },
            "HIGH_RISK_DIABETES_MELLITUS": {
                "condition": "Diabetes Mellitus",
                "actions": [
                    "HbA1c testing within 2 weeks",
                    "Review glucose control",
                    "Assess medication adherence"
                ],
                "urgency": "medium"
            },
            "HIGH_RISK_COPD": {
                "condition": "COPD",
                "actions": [
                    "Pulmonology follow-up within 1 week",
                    "Review inhaler technique",
                    "Assess oxygen requirements"
                ],
                "urgency": "high"
            },
            "POLYPHARMACY": {
                "condition": "Multiple Medications",
                "actions": [
                    "Medication reconciliation",
                    "Check for drug interactions",
                    "Assess adherence barriers"
                ],
                "urgency": "medium"
            },
            "POST_DISCHARGE_HIGH_RISK": {
                "condition": "Recent Discharge",
                "actions": [
                    "Schedule follow-up within 48-72 hours",
                    "Verify medication understanding",
                    "Review discharge instructions"
                ],
                "urgency": "high"
            }
        }
        
        if os.path.exists(file):
            with open(file, 'r') as f:
                return json.load(f)
        else:
            with open(file, 'w') as f:
                json.dump(default, f, indent=4)
            print(f"✓ Created: {file}")
            return default
    
    def load_patients(self):
        """Load or create patients_data.csv"""
        file = 'patients_data.csv'
        default = [
            {
                'id': 1, 'name': 'John Davis', 'age': 67, 'mrn': 'MRN-10234',
                'lab_result': 78, 'procedure_priority': 3,
                'clinical_note': 'Patient presents with shortness of breath and fatigue. History of CHF and diabetes. Currently on Metformin, Lisinopril, and Furosemide. Recent ED visit 5 days ago for volume overload.'
            },
            {
                'id': 2, 'name': 'Sarah Martinez', 'age': 54, 'mrn': 'MRN-10235',
                'lab_result': 45, 'procedure_priority': 2,
                'clinical_note': 'Patient with hypertension and diabetes mellitus type 2. Stable on current medications. No acute complaints.'
            },
            {
                'id': 3, 'name': 'Robert Chen', 'age': 72, 'mrn': 'MRN-10236',
                'lab_result': 85, 'procedure_priority': 4,
                'clinical_note': 'COPD exacerbation with chest pain. Multiple comorbidities including diabetes, hypertension, and heart failure. On insulin, Lisinopril, Metoprolol, and inhalers.'
            },
            {
                'id': 4, 'name': 'Maria Garcia', 'age': 58, 'mrn': 'MRN-10237',
                'lab_result': 62, 'procedure_priority': 2,
                'clinical_note': 'Type 2 diabetes with poor glycemic control. HbA1c 9.2%. Patient reports medication non-adherence.'
            }
        ]
        
        if os.path.exists(file):
            df = pd.read_csv(file)
            return df.to_dict('records')
        else:
            df = pd.DataFrame(default)
            df.to_csv(file, index=False)
            print(f"✓ Created: {file}")
            return default
    
    def load_ml_model(self):
        """Load trained ML model if available"""
        path = self.config['ml_model']['model_path']
        if os.path.exists(path) and ML_AVAILABLE:
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None


# ============================================================================
# AGENT 5: AUDIT/LOGGING AGENT
# ============================================================================

class AgentLogger:
    """Tracks all agent activities with timestamps"""
    
    def __init__(self):
        self.logs = []
        os.makedirs('logs', exist_ok=True)
        self.log_file = f"logs/trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def add_log(self, agent, message, status):
        entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'agent': agent,
            'message': message,
            'status': status
        }
        self.logs.append(entry)
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{entry['timestamp']}] [{status.upper()}] {agent}: {message}\n")
        
        return entry


# ============================================================================
# AGENT 1: RISK SCORING AGENT
# ============================================================================

class RiskScoringAgent:
    """Integrates ML model from Problem 1 for risk prediction"""
    
    def __init__(self, logger, config_mgr):
        self.logger = logger
        self.config = config_mgr.config['ml_model']
        self.ml_model = config_mgr.ml_model
        self.thresholds = self.config['risk_thresholds']
        self.weights = self.config['weights']
        self.delay = config_mgr.config['agent_delays']['risk_scoring']
    
    def predict_risk(self, lab_result, procedure_priority):
        self.logger.add_log('Risk Scoring Agent', 
            f'Analyzing data: lab={lab_result}, procedure={procedure_priority}', 
            'processing')
        time.sleep(self.delay)
        
        try:
            if self.ml_model and ML_AVAILABLE:
                # Use real ML model
                features = np.array([[lab_result, procedure_priority]])
                risk_proba = self.ml_model.predict_proba(features)[0]
                risk_score = risk_proba[0] * 100
                confidence = max(risk_proba) * 100
            else:
                # Simulate XGBoost prediction (Problem 1 logic)
                risk_score = min(95, max(5, 
                    (lab_result * self.weights['lab_result']) + 
                    (procedure_priority * self.weights['procedure_priority']) + 
                    (np.random.random() * 15 - 7.5)
                ))
                confidence = 85 + np.random.random() * 10
            
            risk_score = round(risk_score, 1)
            confidence = round(confidence, 1)
            
            if risk_score >= self.thresholds['high']:
                category = 'HIGH'
            elif risk_score >= self.thresholds['medium']:
                category = 'MEDIUM'
            else:
                category = 'LOW'
            
            result = {
                'risk_score': risk_score,
                'risk_category': category,
                'confidence': confidence
            }
            
            self.logger.add_log('Risk Scoring Agent', 
                f'Prediction: {risk_score}% ({category}), Conf: {confidence}%', 
                'success')
            return {'success': True, 'data': result}
            
        except Exception as e:
            self.logger.add_log('Risk Scoring Agent', f'Error: {str(e)}', 'error')
            return {'success': False, 'error': str(e)}


# ============================================================================
# AGENT 2: CLINICAL UNDERSTANDING AGENT
# ============================================================================

class ClinicalUnderstandingAgent:
    """Integrates LLM from Problem 2 for clinical extraction"""
    
    def __init__(self, logger, config_mgr):
        self.logger = logger
        self.config = config_mgr.config['llm_extractor']
        self.api_key = os.getenv(self.config['api_key_env'], '')
        self.delay = config_mgr.config['agent_delays']['clinical_understanding']
    
    def extract_clinical_info(self, clinical_note):
        self.logger.add_log('Clinical Understanding Agent', 
            f'Extracting from note ({len(clinical_note)} chars)', 
            'processing')
        time.sleep(self.delay)
        
        try:
            if self.api_key and API_AVAILABLE:
                extracted = self._call_openrouter(clinical_note)
            else:
                extracted = self._rule_based_extraction(clinical_note)
            
            self.logger.add_log('Clinical Understanding Agent',
                f'Found: {len(extracted["conditions"])} conditions, '
                f'{len(extracted["medications"])} meds, '
                f'{len(extracted["symptoms"])} symptoms', 
                'success')
            return {'success': True, 'data': extracted}
            
        except Exception as e:
            self.logger.add_log('Clinical Understanding Agent', 
                f'Error: {str(e)}', 'error')
            return {'success': False, 'error': str(e)}
    
    def _call_openrouter(self, note):
        """Integration with Problem 2 OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Clinical-Extraction"
        }
        
        payload = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system", 
                    "content": "Extract conditions, medications, and symptoms from clinical notes. Return JSON only with keys: conditions, medications, symptoms."
                },
                {"role": "user", "content": f"Extract from: {note}"}
            ],
            "temperature": self.config['temperature']
        }
        
        response = requests.post(self.config['api_url'], 
            headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        # Ensure required keys
        data.setdefault('conditions', [])
        data.setdefault('medications', [])
        data.setdefault('symptoms', [])
        
        return data
    
    def _rule_based_extraction(self, note):
        """Fallback: Rule-based extraction matching Problem 2 patterns"""
        note_lower = note.lower()
        
        conditions = []
        for keyword, condition in [
            ('diabetes', 'Diabetes Mellitus'),
            ('hypertension', 'Hypertension'),
            ('chf', 'Congestive Heart Failure'),
            ('heart failure', 'Congestive Heart Failure'),
            ('copd', 'COPD'),
            ('cad', 'Coronary Artery Disease'),
            ('ckd', 'Chronic Kidney Disease')
        ]:
            if keyword in note_lower and condition not in conditions:
                conditions.append(condition)
        
        medications = []
        for med in ['Metformin', 'Insulin', 'Lisinopril', 'Furosemide',
                    'Metoprolol', 'Atorvastatin', 'Amlodipine']:
            if med.lower() in note_lower:
                medications.append(med)
        
        symptoms = []
        for keyword, symptom in [
            ('shortness of breath', 'Dyspnea'),
            ('dyspnea', 'Dyspnea'),
            ('chest pain', 'Chest Pain'),
            ('fatigue', 'Fatigue')
        ]:
            if keyword in note_lower and symptom not in symptoms:
                symptoms.append(symptom)
        
        return {
            'conditions': conditions,
            'medications': medications,
            'symptoms': symptoms
        }


# ============================================================================
# AGENT 3: GUIDELINE REASONING AGENT
# ============================================================================

class GuidelineReasoningAgent:
    """Matches patient state to clinical guidelines"""
    
    def __init__(self, logger, config_mgr):
        self.logger = logger
        self.guidelines = config_mgr.guidelines
        self.delay = config_mgr.config['agent_delays']['guideline_reasoning']
    
    def match_guidelines(self, risk_data, clinical_data):
        self.logger.add_log('Guideline Reasoning Agent',
            f'Matching {len(clinical_data["conditions"])} conditions',
            'processing')
        time.sleep(self.delay)
        
        guidelines = []
        care_gaps = []
        
        # High risk guidelines
        if risk_data['risk_category'] == 'HIGH':
            for condition in clinical_data['conditions']:
                key = f"HIGH_RISK_{condition.upper().replace(' ', '_')}"
                if key in self.guidelines:
                    guidelines.append(self.guidelines[key])
            
            if 'POST_DISCHARGE_HIGH_RISK' in self.guidelines:
                guidelines.append(self.guidelines['POST_DISCHARGE_HIGH_RISK'])
        
        # Polypharmacy
        if len(clinical_data['medications']) >= 3:
            if 'POLYPHARMACY' in self.guidelines:
                guidelines.append(self.guidelines['POLYPHARMACY'])
            care_gaps.append(f'Polypharmacy: {len(clinical_data["medications"])} medications')
        
        # Care gaps
        if 'Diabetes Mellitus' in clinical_data['conditions']:
            has_med = any(m.lower() in ['metformin', 'insulin'] 
                         for m in clinical_data['medications'])
            if not has_med:
                care_gaps.append('Diabetes without glucose-lowering medication')
        
        if 'Hypertension' in clinical_data['conditions']:
            has_med = any(m.lower() in ['lisinopril', 'amlodipine', 'metoprolol'] 
                         for m in clinical_data['medications'])
            if not has_med:
                care_gaps.append('Hypertension without antihypertensive therapy')
        
        # Remove duplicates
        unique = []
        seen = set()
        for g in guidelines:
            if g['condition'] not in seen:
                seen.add(g['condition'])
                unique.append(g)
        
        self.logger.add_log('Guideline Reasoning Agent',
            f'Matched {len(unique)} guidelines, {len(care_gaps)} care gaps',
            'success')
        
        return {
            'success': True,
            'data': {
                'applicable_guidelines': unique,
                'care_gaps': care_gaps
            }
        }


# ============================================================================
# AGENT 4: DECISION AGENT
# ============================================================================

class DecisionAgent:
    """Synthesizes all inputs into actionable recommendations"""
    
    def __init__(self, logger, config_mgr):
        self.logger = logger
        self.delay = config_mgr.config['agent_delays']['decision_making']
    
    def synthesize_recommendations(self, risk_data, clinical_data, guideline_data):
        self.logger.add_log('Decision Agent', 
            'Synthesizing recommendations', 'processing')
        time.sleep(self.delay)
        
        recommendations = []
        priorities = {'high': [], 'medium': [], 'low': []}
        
        # Guideline-based recommendations
        for guideline in guideline_data['applicable_guidelines']:
            for action in guideline['actions']:
                rec = {
                    'action': action,
                    'rationale': f"Guideline: {guideline['condition']}",
                    'urgency': guideline['urgency'],
                    'source': 'clinical_guideline'
                }
                recommendations.append(rec)
                priorities[guideline['urgency']].append(action)
        
        # Risk-based recommendations
        if risk_data['risk_category'] == 'HIGH':
            rec = {
                'action': 'Consider hospital admission or observation unit',
                'rationale': f"ML Model: {risk_data['risk_score']}% readmission risk",
                'urgency': 'high',
                'source': 'ml_model'
            }
            recommendations.append(rec)
            priorities['high'].append(rec['action'])
        
        # Symptom-based recommendations
        if 'Dyspnea' in clinical_data.get('symptoms', []) and \
           'Congestive Heart Failure' in clinical_data.get('conditions', []):
            rec = {
                'action': 'Urgent volume status assessment - consider IV diuresis',
                'rationale': 'Clinical: Dyspnea in CHF patient',
                'urgency': 'high',
                'source': 'clinical_reasoning'
            }
            recommendations.append(rec)
            priorities['high'].append(rec['action'])
        
        # Calculate confidence
        ml_conf = risk_data['confidence'] * 0.7
        bonus = 20 if guideline_data['applicable_guidelines'] else 10
        overall_confidence = min(95, ml_conf + bonus)
        
        self.logger.add_log('Decision Agent',
            f'Generated {len(recommendations)} recommendations (Conf: {overall_confidence:.1f}%)',
            'success')
        
        return {
            'success': True,
            'data': {
                'recommendations': recommendations,
                'priorities': priorities,
                'overall_confidence': overall_confidence
            }
        }


# ============================================================================
# USER INTERFACE
# ============================================================================

class PatientRiskAgentUI:
    """Main Tkinter Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Patient Risk Review System v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f3f4f6")
        
        # Initialize system
        try:
            self.config_mgr = ConfigManager()
            self.logger = AgentLogger()
            self.risk_agent = RiskScoringAgent(self.logger, self.config_mgr)
            self.clinical_agent = ClinicalUnderstandingAgent(self.logger, self.config_mgr)
            self.guideline_agent = GuidelineReasoningAgent(self.logger, self.config_mgr)
            self.decision_agent = DecisionAgent(self.logger, self.config_mgr)
            
            self.patients = self.config_mgr.patients
            self.selected_patient = None
            self.is_processing = False
            self.analysis_result = None
            
            self.setup_ui()
            self.show_startup()
            
        except Exception as e:
            messagebox.showerror("Init Error", f"Failed to start:\n{str(e)}")
            raise
    
    def show_startup(self):
        """Show system status"""
        ml = "✓ Loaded" if self.config_mgr.ml_model else "⚠ Simulated"
        api = "✓ Available" if self.clinical_agent.api_key else "⚠ Simulated"
        
        info = f"""System Ready

Files:
✓ config.json
✓ clinical_guidelines.json  
✓ patients_data.csv ({len(self.patients)} patients)

Agents:
✓ All 5 agents initialized

Integration:
  ML Model: {ml}
  LLM API: {api}

Ready to process patients."""
        
        messagebox.showinfo("System Status", info)
    
    def setup_ui(self):
        """Build UI"""
        # Header
        header = tk.Frame(self.root, bg="#1e40af", height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="🛡 Autonomous Patient Risk Review System",
                font=("Arial", 20, "bold"), fg="white", bg="#1e40af").pack(pady=25)
        
        # Main container
        main = tk.Frame(self.root, bg="#f3f4f6")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Patient queue
        left = tk.Frame(main, bg="white", relief=tk.RAISED, bd=1, width=300)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left.pack_propagate(False)
        
        tk.Label(left, text="📋 Patient Queue", 
                font=("Arial", 12, "bold"), bg="white").pack(pady=10)
        
        self.patient_list = tk.Listbox(left, font=("Arial", 10))
        self.patient_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.patient_list.bind('<<ListboxSelect>>', self.on_patient_select)
        
        for p in self.patients:
            self.patient_list.insert(tk.END, f"{p['name']} ({p['mrn']})")
        
        # Right: Results
        right = tk.Frame(main, bg="#f3f4f6")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = tk.Frame(right, bg="#dbeafe", relief=tk.RAISED, bd=1)
        self.status_frame.pack(fill=tk.X)
        self.status_label = tk.Label(self.status_frame, 
            text="Select a patient to begin analysis",
            font=("Arial", 10), bg="#dbeafe", fg="#1e40af")
        self.status_label.pack(pady=10)
        
        # Tabs
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Summary
        tab1 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab1, text="Risk & Clinical Summary")
        
        # Risk display
        risk_frame = tk.LabelFrame(tab1, text=" Risk Assessment ",
            font=("Arial", 11, "bold"), bg="white", padx=10, pady=10)
        risk_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.risk_score = tk.Label(risk_frame, text="--", 
            font=("Arial", 36, "bold"), bg="white", fg="#6b7280")
        self.risk_score.pack()
        
        self.risk_category = tk.Label(risk_frame, text="No Analysis",
            font=("Arial", 12), bg="white", fg="#6b7280")
        self.risk_category.pack()
        
        self.confidence = tk.Label(risk_frame, text="",
            font=("Arial", 9), bg="white", fg="#6b7280")
        self.confidence.pack(pady=5)
        
        # Clinical facts
        facts_frame = tk.LabelFrame(tab1, text=" Clinical Facts ",
            font=("Arial", 11, "bold"), bg="white", padx=10, pady=10)
        facts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.facts_text = scrolledtext.ScrolledText(facts_frame, 
            height=15, font=("Arial", 10), wrap=tk.WORD, bg="#f9fafb")
        self.facts_text.pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Recommendations
        tab2 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab2, text="Recommendations")
        
        self.rec_text = scrolledtext.ScrolledText(tab2, 
            font=("Arial", 10), wrap=tk.WORD, bg="white")
        self.rec_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.rec_text.tag_config("high", foreground="#dc2626", font=("Arial", 10, "bold"))
        self.rec_text.tag_config("medium", foreground="#ea580c", font=("Arial", 10, "bold"))
        self.rec_text.tag_config("header", font=("Arial", 11, "bold"))
        
        # Tab 3: Agent Logs
        tab3 = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab3, text="Agent Execution Trace")
        
        self.log_text = scrolledtext.ScrolledText(tab3, 
            font=("Courier", 9), wrap=tk.WORD, bg="#f9fafb")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text.tag_config("success", foreground="#16a34a")
        self.log_text.tag_config("error", foreground="#dc2626")
        self.log_text.tag_config("processing", foreground="#2563eb")
    
    def on_patient_select(self, event):
        """Handle patient selection"""
        if self.is_processing:
            messagebox.showwarning("Busy", "Please wait for current analysis")
            return
        
        sel = self.patient_list.curselection()
        if sel:
            self.selected_patient = self.patients[sel[0]]
            threading.Thread(target=self.process_patient, daemon=True).start()
    
    def process_patient(self):
        """Orchestrate all agents"""
        self.is_processing = True
        self.logger.logs = []
        
        self.root.after(0, self.show_status, "Initializing agents...")
        self.root.after(0, self.clear_results)
        
        try:
            p = self.selected_patient
            
            self.logger.add_log('Orchestrator', 
                f'Starting analysis: {p["name"]}', 'processing')
            self.root.after(0, self.update_logs)
            
            # Agent 1: Risk Scoring
            self.root.after(0, self.show_status, "Risk Scoring Agent...")
            risk_result = self.risk_agent.predict_risk(
                p['lab_result'], p['procedure_priority'])
            self.root.after(0, self.update_logs)
            if not risk_result['success']:
                raise Exception('Risk scoring failed')
            
            # Agent 2: Clinical Understanding
            self.root.after(0, self.show_status, "Clinical Understanding Agent...")
            clinical_result = self.clinical_agent.extract_clinical_info(
                p['clinical_note'])
            self.root.after(0, self.update_logs)
            if not clinical_result['success']:
                raise Exception('Clinical extraction failed')
            
            # Agent 3: Guideline Reasoning
            self.root.after(0, self.show_status, "Guideline Reasoning Agent...")
            guideline_result = self.guideline_agent.match_guidelines(
                risk_result['data'], clinical_result['data'])
            self.root.after(0, self.update_logs)
            if not guideline_result['success']:
                raise Exception('Guideline matching failed')
            
            # Agent 4: Decision Making
            self.root.after(0, self.show_status, "Decision Agent...")
            decision_result = self.decision_agent.synthesize_recommendations(
                risk_result['data'], clinical_result['data'], 
                guideline_result['data'])
            self.root.after(0, self.update_logs)
            if not decision_result['success']:
                raise Exception('Decision synthesis failed')
            
            # Store results
            self.analysis_result = {
                'patient': p,
                'risk': risk_result['data'],
                'clinical': clinical_result['data'],
                'guidelines': guideline_result['data'],
                'decisions': decision_result['data']
            }
            
            self.logger.add_log('Orchestrator', 'Analysis complete', 'success')
            self.root.after(0, self.show_status, "✓ Analysis Complete")
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.logger.add_log('Orchestrator', f'Failed: {str(e)}', 'error')
            self.root.after(0, self.show_status, f"✗ Error: {str(e)}")
            self.root.after(0, self.update_logs)
            messagebox.showerror("Analysis Failed", str(e))
        finally:
            self.is_processing = False
    
    def show_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
    
    def clear_results(self):
        """Clear all display areas"""
        self.risk_score.config(text="--", fg="#6b7280")
        self.risk_category.config(text="Analyzing...", fg="#6b7280")
        self.confidence.config(text="")
        
        self.facts_text.config(state=tk.NORMAL)
        self.facts_text.delete(1.0, tk.END)
        self.facts_text.config(state=tk.DISABLED)
        
        self.rec_text.config(state=tk.NORMAL)
        self.rec_text.delete(1.0, tk.END)
        self.rec_text.config(state=tk.DISABLED)
    
    def update_logs(self):
        """Update agent log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        
        for log in self.logger.logs:
            line = f"[{log['timestamp']}] {log['agent']}\n  {log['message']}\n\n"
            self.log_text.insert(tk.END, line, log['status'])
        
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def display_results(self):
        """Display complete analysis results"""
        if not self.analysis_result:
            return
        
        r = self.analysis_result
        
        # Risk score display
        score = r['risk']['risk_score']
        category = r['risk']['risk_category']
        
        color_map = {
            'HIGH': '#dc2626',
            'MEDIUM': '#ea580c',
            'LOW': '#16a34a'
        }
        
        self.risk_score.config(text=f"{score}%", fg=color_map[category])
        self.risk_category.config(text=f"{category} RISK", fg=color_map[category])
        self.confidence.config(
            text=f"Model Confidence: {r['risk']['confidence']}%",
            fg="#6b7280"
        )
        
        # Clinical facts
        self.facts_text.config(state=tk.NORMAL)
        self.facts_text.delete(1.0, tk.END)
        
        self.facts_text.insert(tk.END, f"PATIENT: {r['patient']['name']}\n", "header")
        self.facts_text.insert(tk.END, f"MRN: {r['patient']['mrn']} | Age: {r['patient']['age']}\n\n")
        
        self.facts_text.insert(tk.END, "CONDITIONS:\n", "header")
        if r['clinical']['conditions']:
            for cond in r['clinical']['conditions']:
                self.facts_text.insert(tk.END, f"  • {cond}\n")
        else:
            self.facts_text.insert(tk.END, "  None documented\n")
        
        self.facts_text.insert(tk.END, "\nMEDICATIONS:\n", "header")
        if r['clinical']['medications']:
            for med in r['clinical']['medications']:
                self.facts_text.insert(tk.END, f"  • {med}\n")
        else:
            self.facts_text.insert(tk.END, "  None documented\n")
        
        self.facts_text.insert(tk.END, "\nSYMPTOMS:\n", "header")
        if r['clinical']['symptoms']:
            for symptom in r['clinical']['symptoms']:
                self.facts_text.insert(tk.END, f"  • {symptom}\n")
        else:
            self.facts_text.insert(tk.END, "  None documented\n")
        
        if r['guidelines']['care_gaps']:
            self.facts_text.insert(tk.END, "\nCARE GAPS:\n", "header")
            for gap in r['guidelines']['care_gaps']:
                self.facts_text.insert(tk.END, f"  ⚠ {gap}\n")
        
        self.facts_text.config(state=tk.DISABLED)
        
        # Recommendations
        self.rec_text.config(state=tk.NORMAL)
        self.rec_text.delete(1.0, tk.END)
        
        self.rec_text.insert(tk.END, 
            f"CLINICAL DECISION SUPPORT RECOMMENDATIONS\n", "header")
        self.rec_text.insert(tk.END, 
            f"Overall Confidence: {r['decisions']['overall_confidence']:.1f}%\n\n")
        
        # Group by urgency
        if r['decisions']['priorities']['high']:
            self.rec_text.insert(tk.END, "🔴 HIGH PRIORITY ACTIONS:\n", "high")
            for i, action in enumerate(r['decisions']['priorities']['high'], 1):
                self.rec_text.insert(tk.END, f"{i}. {action}\n")
            self.rec_text.insert(tk.END, "\n")
        
        if r['decisions']['priorities']['medium']:
            self.rec_text.insert(tk.END, "🟠 MEDIUM PRIORITY ACTIONS:\n", "medium")
            for i, action in enumerate(r['decisions']['priorities']['medium'], 1):
                self.rec_text.insert(tk.END, f"{i}. {action}\n")
            self.rec_text.insert(tk.END, "\n")
        
        # Detailed recommendations with rationale
        self.rec_text.insert(tk.END, "\nDETAILED RATIONALE:\n", "header")
        for i, rec in enumerate(r['decisions']['recommendations'], 1):
            tag = rec['urgency']
            self.rec_text.insert(tk.END, f"\n{i}. {rec['action']}\n", tag)
            self.rec_text.insert(tk.END, f"   Rationale: {rec['rationale']}\n")
            self.rec_text.insert(tk.END, f"   Source: {rec['source']}\n")
        
        self.rec_text.config(state=tk.DISABLED)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Application entry point"""
    root = tk.Tk()
    app = PatientRiskAgentUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()