import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from datetime import datetime
import sqlite3
import os
import yaml

class TemporalFilter:
    """Filtro temporal para reducir falsos positivos mediante suavizado"""
    def __init__(self, window_size=5, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}
    
    def update(self, violation_type, is_detected):
        """Actualiza el historial de detección para un tipo de violación"""
        if violation_type not in self.history:
            self.history[violation_type] = deque(maxlen=self.window_size)
        
        self.history[violation_type].append(1 if is_detected else 0)
    
    def is_valid(self, violation_type):
        """Verifica si la violación es válida según el umbral temporal"""
        if violation_type not in self.history:
            return False
        
        # Reducido a 2 frames mínimo en lugar de 3
        if len(self.history[violation_type]) < 2:
            return False
        
        detection_rate = sum(self.history[violation_type]) / len(self.history[violation_type])
        return detection_rate >= self.threshold
    
    def reset(self, violation_type):
        """Reinicia el historial de un tipo de violación"""
        if violation_type in self.history:
            self.history[violation_type].clear()

class VideoClipRecorder:
    """Grabador de clips de video con buffer circular"""
    def __init__(self, clip_duration=5.0, fps=30, codec='avc1', resolution=(1280, 720), format='.mp4'):
        self.clip_duration = clip_duration
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.format = format
        self.buffer_size = int(clip_duration * fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.recording = False
        self.frames_after_trigger = 0
        self.max_frames_after = int(fps * 2.5)
    
    def add_frame(self, frame):
        """Añade un frame al buffer circular"""
        self.frame_buffer.append(frame.copy())
    
    def start_recording(self):
        """Inicia la grabación de un clip"""
        self.recording = True
        self.frames_after_trigger = 0
    
    def save_clip(self, output_path):
        """Guarda el clip de video desde el buffer"""
        if len(self.frame_buffer) == 0:
            print(f"Error: Buffer vacio, no se puede guardar {output_path}")
            return False
        
        if self.format == '.webm' and output_path.endswith('.mp4'):
            output_path = output_path.replace('.mp4', '.webm')
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, self.resolution)
            
            if not out.isOpened():
                print(f"Error: No se pudo abrir VideoWriter para {output_path}")
                return False
            
            frames_written = 0
            for frame in self.frame_buffer:
                if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                    frame = cv2.resize(frame, self.resolution)
                out.write(frame)
                frames_written += 1
            
            out.release()
            print(f"Clip guardado: {output_path} ({frames_written} frames)")
            return True
            
        except Exception as e:
            print(f"Error guardando clip: {e}")
            return False

class AdvancedDriverMonitoring:
    def __init__(self, config_path='config.yaml'):
        print("Inicializando sistema avanzado...")
        
        self.config = self.load_config(config_path)
        
        # Modelos YOLO11
        self.yolo_model = YOLO(self.config['models']['detection'])
        self.pose_model = YOLO(self.config['models']['pose'])
        
        # Filtro temporal MÁS PERMISIVO
        self.temporal_filter = TemporalFilter(
            window_size=self.config['temporal_filter']['window_size'],
            threshold=self.config['temporal_filter']['validation_threshold']
        )
        
        # Grabador de clips
        self.video_recorder = VideoClipRecorder(
            clip_duration=self.config['video_recording']['clip_duration'],
            fps=self.config['video_recording']['fps'],
            codec=self.config['video_recording']['codec'],
            resolution=tuple(self.config['video_recording']['resolution']),
            format=self.config['video_recording'].get('format', '.mp4')
        )
        
        # Buffers temporales
        self.behavior_history = deque(maxlen=90)
        self.hand_position_history = deque(maxlen=60)
        self.gaze_history = deque(maxlen=30)
        
        # Estado del sistema
        self.current_violations = {}
        self.violation_start_times = {}
        self.alert_cooldown = {}
        self.frame_count = 0
        self.fps = 0
        
        # DEBUG
        self.debug_mode = True
        
        # Base de datos
        self.init_database()
        
        # Crear carpetas por nivel de riesgo
        os.makedirs('captures', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        os.makedirs('clips/criticas', exist_ok=True)
        os.makedirs('clips/alto_riesgo', exist_ok=True)
        os.makedirs('clips/sospechosas', exist_ok=True)
        
        print("Sistema avanzado inicializado\n")
        self.print_safety_standards()
    
    def load_config(self, config_path):
        """Carga la configuración desde archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Configuracion cargada desde: {config_path}")
            return config
        except FileNotFoundError:
            print(f"ADVERTENCIA: No se encontro {config_path}, usando configuracion por defecto")
            return self.get_default_config()
        except Exception as e:
            print(f"Error cargando configuracion: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Retorna configuración por defecto MUCHO MÁS SENSIBLE"""
        return {
            'models': {
                'detection': 'yolo11n.pt',
                'pose': 'yolo11n-pose.pt',
                'detection_conf': 0.45,
                'pose_conf': 0.5
            },
            'zones': {
                'steering_wheel': {'y_range': [0.5, 0.95], 'x_range': [0.2, 0.8]},
                'face_region': {'y_range': [0.15, 0.55], 'x_range': [0.3, 0.7]}
            },
            'thresholds': {
                'hand_steering_distance': 250,
                'confidence': 0.5,
                'object_detection': 0.3,  # Más sensible
                'object_near_hand': 120,
                'gaze_deviation_sospechoso': 30,  # MÁS SENSIBLE
                'gaze_deviation_alto_riesgo': 50   # MÁS SENSIBLE
            },
            'temporal_filter': {
                'window_size': 5,
                'validation_threshold': 0.5  # Reducido de 0.6 a 0.5
            },
            'video_recording': {
                'clip_duration': 5.0,
                'fps': 30,
                'codec': 'avc1',
                'resolution': [1280, 720]
            },
            'safety_standards': {
                'CRITICO': {
                    'ambas_manos_fuera_volante': {'weight': 1.0, 'duration': 0.8},  # MÁS RÁPIDO
                    'conductor_ausente': {'weight': 1.0, 'duration': 0.2},  # MÁS RÁPIDO
                    'multiples_personas': {'weight': 0.9, 'duration': 1.0}  # MÁS RÁPIDO
                },
                'ALTO_RIESGO': {
                    'una_mano_fuera_volante': {'weight': 0.75, 'duration': 1.5},  # MÁS RÁPIDO
                    'uso_celular': {'weight': 0.85, 'duration': 0.8},  # MÁS RÁPIDO
                    'mirada_muy_desviada': {'weight': 0.80, 'duration': 2.0}  # MÁS RÁPIDO
                },
                'SOSPECHOSO': {
                    'objeto_detectado': {'weight': 0.60, 'duration': 1.5},  # MÁS RÁPIDO
                    'mirada_desviada': {'weight': 0.55, 'duration': 2.0}  # MÁS RÁPIDO
                }
            }
        }
    
    def print_safety_standards(self):
        print("ESTANDARES DE CONDUCCION SEGURA:")
        print("="*70)
        for level, violations in self.config['safety_standards'].items():
            print(f"\n[{level}]:")
            for violation, params in violations.items():
                print(f"   - {violation.replace('_', ' ').title()}")
                print(f"     Peso: {params['weight']:.0%} | Duracion: {params['duration']}s")
        print("\n" + "="*70 + "\n")
    
    def init_database(self):
        self.conn = sqlite3.connect('driver_monitoring_advanced.db')
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                risk_level TEXT,
                violation_type TEXT,
                duration REAL,
                confidence REAL,
                hands_on_wheel INTEGER,
                person_count INTEGER,
                objects_detected TEXT,
                image_path TEXT,
                video_clip_path TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TEXT,
                session_end TEXT,
                total_frames INTEGER,
                total_violations INTEGER,
                critical_count INTEGER,
                high_risk_count INTEGER,
                suspicious_count INTEGER,
                avg_fps REAL
            )
        ''')
        self.conn.commit()
    
    def analyze_keypoints(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return None
        
        try:
            kp = keypoints[0].cpu().numpy()
            
            analysis = {
                'nose': kp[0][:2],
                'left_eye': kp[1][:2],
                'right_eye': kp[2][:2],
                'left_ear': kp[3][:2],
                'right_ear': kp[4][:2],
                'left_shoulder': kp[5][:2],
                'right_shoulder': kp[6][:2],
                'left_elbow': kp[7][:2],
                'right_elbow': kp[8][:2],
                'left_wrist': kp[9][:2],
                'right_wrist': kp[10][:2],
                'confidence': kp[:, 2]
            }
            
            return analysis
        except Exception as e:
            return None
    
    def detect_gaze_direction(self, keypoints_analysis, frame_shape):
        """Detecta si el conductor está mirando al frente"""
        if not keypoints_analysis:
            return {'looking_forward': True, 'deviation_angle': 0, 'status': 'desconocido', 'severity': 'NORMAL'}
        
        h, w = frame_shape[:2]
        
        nose = keypoints_analysis['nose']
        left_eye = keypoints_analysis['left_eye']
        right_eye = keypoints_analysis['right_eye']
        left_ear = keypoints_analysis['left_ear']
        right_ear = keypoints_analysis['right_ear']
        
        if (nose[0] == 0 or left_eye[0] == 0 or right_eye[0] == 0):
            return {'looking_forward': True, 'deviation_angle': 0, 'status': 'no_detectado', 'severity': 'NORMAL'}
        
        # Calcular centro de los ojos
        eye_center = [(left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2]
        
        # Calcular desviación horizontal
        center_x = w / 2
        nose_deviation = abs(nose[0] - center_x)
        max_deviation = w * 0.20
        
        deviation_percentage = (nose_deviation / max_deviation) * 100 if max_deviation > 0 else 0
        
        # Detectar giro de cabeza usando orejas
        if left_ear[0] > 0 and right_ear[0] > 0:
            ear_distance = abs(left_ear[0] - right_ear[0])
            if ear_distance < w * 0.08:
                deviation_percentage += 20
        
        # Umbrales MÁS SENSIBLES
        threshold_sospechoso = self.config['thresholds'].get('gaze_deviation_sospechoso', 30)
        threshold_alto_riesgo = self.config['thresholds'].get('gaze_deviation_alto_riesgo', 50)
        
        if deviation_percentage < threshold_sospechoso:
            status = 'mirando_frente'
            severity = 'NORMAL'
        elif deviation_percentage < threshold_alto_riesgo:
            status = 'mirada_desviada'
            severity = 'SOSPECHOSO'
        else:
            status = 'mirada_muy_desviada'
            severity = 'ALTO_RIESGO'
        
        looking_forward = deviation_percentage < threshold_sospechoso
        
        return {
            'looking_forward': looking_forward,
            'deviation_angle': deviation_percentage,
            'status': status,
            'severity': severity
        }
    
    def detect_hands_on_steering_wheel(self, keypoints_analysis, frame_shape):
        if not keypoints_analysis:
            return {'left': False, 'right': False, 'count': 0, 'left_pos': [0,0], 'right_pos': [0,0]}
        
        h, w = frame_shape[:2]
        
        y_range = self.config['zones']['steering_wheel']['y_range']
        x_range = self.config['zones']['steering_wheel']['x_range']
        
        steering_y_min = int(h * y_range[0])
        steering_y_max = int(h * y_range[1])
        steering_x_min = int(w * x_range[0])
        steering_x_max = int(w * x_range[1])
        
        left_wrist = keypoints_analysis['left_wrist']
        right_wrist = keypoints_analysis['right_wrist']
        
        def is_hand_on_wheel(wrist, is_left_hand):
            if wrist[0] == 0 and wrist[1] == 0:
                return False
            
            if not (steering_y_min < wrist[1] < steering_y_max):
                return False
            
            if not (steering_x_min < wrist[0] < steering_x_max):
                return False
            
            center_x = w / 2
            if is_left_hand:
                return wrist[0] > center_x * 0.7
            else:
                return wrist[0] < center_x * 1.3
        
        left_on_wheel = is_hand_on_wheel(left_wrist, True)
        right_on_wheel = is_hand_on_wheel(right_wrist, False)
        
        return {
            'left': left_on_wheel,
            'right': right_on_wheel,
            'count': int(left_on_wheel) + int(right_on_wheel),
            'left_pos': left_wrist,
            'right_pos': right_wrist
        }
    
    def detect_suspicious_objects(self, yolo_results, hands_info):
        """Detecta objetos: celular=ALTO_RIESGO, otros=SOSPECHOSO"""
        detected = []
        
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Lista de objetos a detectar
                detectable_objects = [
                    'cell phone', 'bottle', 'cup', 'wine glass', 'fork', 'knife', 
                    'spoon', 'book', 'laptop', 'remote', 'banana', 'apple', 
                    'sandwich', 'hot dog', 'pizza', 'donut', 'cake'
                ]
                
                if class_name in detectable_objects and confidence > self.config['thresholds']['object_detection']:
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if class_name == 'cell phone':
                        risk_level = 'ALTO_RIESGO'
                    else:
                        risk_level = 'SOSPECHOSO'
                    
                    near_hand = False
                    if hands_info:
                        obj_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                        for hand_pos in [hands_info['left_pos'], hands_info['right_pos']]:
                            if hand_pos[0] > 0:
                                dist = np.sqrt((obj_center[0]-hand_pos[0])**2 + 
                                             (obj_center[1]-hand_pos[1])**2)
                                if dist < self.config['thresholds']['object_near_hand']:
                                    near_hand = True
                                    break
                    
                    detected.append({
                        'object': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'risk_level': risk_level,
                        'near_hand': near_hand
                    })
        
        return detected
    
    def evaluate_violations(self, hands_info, gaze_info, objects, person_count):
        violations = []
        current_time = time.time()
        raw_violations = []
        
        # 1. Conductor ausente (0 personas)
        if person_count == 0:
            raw_violations.append('conductor_ausente')
        
        # 2. Múltiples personas (>1)
        if person_count > 1:
            raw_violations.append('multiples_personas')
        
        # 3. Ambas manos fuera del volante
        if hands_info['count'] == 0:
            raw_violations.append('ambas_manos_fuera_volante')
        
        # 4. Una mano fuera del volante
        if hands_info['count'] == 1:
            raw_violations.append('una_mano_fuera_volante')
        
        # 5. Uso de celular (ALTO_RIESGO)
        has_cellphone = False
        for obj in objects:
            if obj['object'] == 'cell phone':
                raw_violations.append('uso_celular')
                has_cellphone = True
                break
        
        # 6. Otros objetos detectados (SOSPECHOSO)
        if not has_cellphone and len(objects) > 0:
            raw_violations.append('objeto_detectado')
        
        # 7. Mirada desviada según severidad
        if gaze_info['severity'] == 'SOSPECHOSO':
            raw_violations.append('mirada_desviada')
        elif gaze_info['severity'] == 'ALTO_RIESGO':
            raw_violations.append('mirada_muy_desviada')
        
        # DEBUG: Mostrar violaciones detectadas
        if self.debug_mode and raw_violations and self.frame_count % 30 == 0:
            print(f"[DEBUG] Raw violations: {raw_violations}")
            print(f"[DEBUG] Gaze severity: {gaze_info['severity']}, deviation: {gaze_info['deviation_angle']:.1f}%")
            print(f"[DEBUG] Hands count: {hands_info['count']}")
            print(f"[DEBUG] Objects: {len(objects)}")
        
        # Aplicar filtro temporal
        all_violation_types = set()
        for level_violations in self.config['safety_standards'].values():
            all_violation_types.update(level_violations.keys())
        
        for v_type in all_violation_types:
            is_detected = v_type in raw_violations
            self.temporal_filter.update(v_type, is_detected)
        
        # Validar violaciones
        for v_type in raw_violations:
            is_valid = self.temporal_filter.is_valid(v_type)
            
            # DEBUG
            if self.debug_mode and self.frame_count % 30 == 0:
                history = self.temporal_filter.history.get(v_type, [])
                print(f"[DEBUG] {v_type}: valid={is_valid}, history={list(history)}")
            
            if not is_valid:
                continue
            
            level = None
            params = None
            for lvl, violations_dict in self.config['safety_standards'].items():
                if v_type in violations_dict:
                    level = lvl
                    params = violations_dict[v_type]
                    break
            
            if not level or not params:
                continue
            
            if v_type not in self.violation_start_times:
                self.violation_start_times[v_type] = current_time
            
            duration = current_time - self.violation_start_times[v_type]
            
            # DEBUG
            if self.debug_mode and self.frame_count % 30 == 0:
                print(f"[DEBUG] {v_type}: duration={duration:.2f}s, required={params['duration']}s, level={level}")
            
            if duration >= params['duration']:
                description = self.get_violation_description(
                    v_type, hands_info, gaze_info, objects, person_count
                )
                
                violations.append({
                    'type': v_type,
                    'level': level,
                    'description': description,
                    'confidence': 0.90,
                    'weight': params['weight'],
                    'duration': duration
                })
        
        # Limpiar violaciones inactivas
        current_types = set(raw_violations)
        for v_type in list(self.violation_start_times.keys()):
            if v_type not in current_types:
                del self.violation_start_times[v_type]
                self.temporal_filter.reset(v_type)
        
        return violations
    
    def get_violation_description(self, v_type, hands_info, gaze_info, objects, person_count):
        """Genera descripción legible de la violación"""
        descriptions = {
            'ambas_manos_fuera_volante': 'AMBAS MANOS FUERA DEL VOLANTE',
            'una_mano_fuera_volante': 'Solo una mano en el volante',
            'conductor_ausente': 'CONDUCTOR NO DETECTADO',
            'multiples_personas': f'{person_count} personas en cabina',
            'uso_celular': 'USO DE CELULAR DETECTADO',
            'mirada_desviada': f'Mirada desviada ({gaze_info["deviation_angle"]:.0f}%)',
            'mirada_muy_desviada': f'Mirada muy desviada ({gaze_info["deviation_angle"]:.0f}%)',
            'objeto_detectado': f'{objects[0]["object"] if objects else "Objeto"} detectado'
        }
        return descriptions.get(v_type, v_type.replace('_', ' ').title())
    
    def calculate_risk_score(self, violations):
        if not violations:
            return 'NORMAL', 0.95, 'Conduccion segura'
        
        primary_violation = max(violations, key=lambda v: v['weight'])
        
        level = primary_violation['level']
        confidence = primary_violation['confidence']
        description = primary_violation['description']
        
        if len(violations) > 1:
            description += f" (+{len(violations)-1} mas)"
        
        return level, confidence, description
    
    def generate_alert(self, violations, risk_level):
        if not violations or risk_level == 'NORMAL':
            return None
        
        current_time = time.time()
        primary = max(violations, key=lambda v: v['weight'])
        
        # Cooldown MÁS CORTO
        cooldown_times = {
            'CRITICO': 0.5,  # Muy rápido
            'ALTO_RIESGO': 2.0,  # Reducido
            'SOSPECHOSO': 3.0  # Reducido
        }
        cooldown = cooldown_times.get(risk_level, 2.0)
        
        if risk_level in self.alert_cooldown:
            if current_time - self.alert_cooldown[risk_level] < cooldown:
                return None
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk_level,
            'violations': violations,
            'primary_violation': primary,
            'total_violations': len(violations)
        }
        
        self.alert_cooldown[risk_level] = current_time
        return alert
    
    def save_violation(self, alert, video_path=None, objects_detected=""):
        try:
            primary = alert['primary_violation']
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO violations (timestamp, risk_level, violation_type, 
                                      duration, confidence, image_path, video_clip_path, objects_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['timestamp'],
                alert['risk_level'],
                primary['type'],
                primary['duration'],
                primary['confidence'],
                f"violations/{alert['timestamp'].replace(':', '-')}.jpg",
                video_path,
                objects_detected
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Error guardando violacion: {e}")
    
    def draw_advanced_info(self, frame, hands_info, gaze_info, objects, 
                          violations, risk_level, confidence, description, person_count):
        h, w = frame.shape[:2]
        
        colors = {
            'NORMAL': (0, 255, 0),
            'SOSPECHOSO': (0, 165, 255),
            'ALTO_RIESGO': (0, 140, 255),
            'CRITICO': (0, 0, 255)
        }
        color = colors.get(risk_level, (255, 255, 255))
        
        # Panel principal
        panel_h = 195
        cv2.rectangle(frame, (10, 10), (w - 10, panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_h), color, 3)
        
        cv2.putText(frame, 'SISTEMA AVANZADO DE MONITOREO', (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f'ESTADO: {risk_level}', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        
        cv2.putText(frame, description[:55], (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        y_offset = 130
        cv2.putText(frame, f'Manos: {hands_info["count"]}/2 | Personas: {person_count}', 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if hands_info['count'] == 2 and person_count == 1 else (0, 0, 255), 1)
        
        y_offset += 25
        gaze_color = colors.get(gaze_info['severity'], (0, 255, 0))
        cv2.putText(frame, f'Mirada: {gaze_info["status"]} ({gaze_info["deviation_angle"]:.0f}%)',
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        
        y_offset += 25
        cv2.putText(frame, f'FPS: {self.fps:.1f} | Objetos: {len(objects)}',
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panel de manos
        hand_panel_x = w - 250
        cv2.putText(frame, 'MANOS:', (hand_panel_x, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        left_color = (0, 255, 0) if hands_info['left'] else (100, 100, 100)
        cv2.circle(frame, (hand_panel_x + 50, 55), 15, left_color, -1)
        cv2.putText(frame, 'I', (hand_panel_x + 45, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        right_color = (0, 255, 0) if hands_info['right'] else (100, 100, 100)
        cv2.circle(frame, (hand_panel_x + 100, 55), 15, right_color, -1)
        cv2.putText(frame, 'D', (hand_panel_x + 95, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Violaciones activas
        if violations:
            cv2.putText(frame, f'VIOLACIONES: {len(violations)}',
                       (hand_panel_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            y = 125
            for i, v in enumerate(violations[:3]):
                duration_text = f"{v['duration']:.1f}s"
                cv2.putText(frame, f"{i+1}. {v['type'][:18]} ({duration_text})",
                           (hand_panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)
                y += 20
        
        # Dibujar objetos detectados
        for obj in objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            obj_color = colors.get(obj['risk_level'], (255, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 2)
            
            label = f"{obj['object']} {obj['confidence']:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), obj_color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Alerta según nivel
        if risk_level != 'NORMAL':
            alert_h = 60
            alert_color = colors[risk_level]
            cv2.rectangle(frame, (w//4, h-alert_h-20), (3*w//4, h-20), alert_color, -1)
            
            alert_texts = {
                'CRITICO': '!!! ALERTA CRITICA !!!',
                'ALTO_RIESGO': '!! ALERTA ALTO RIESGO !!',
                'SOSPECHOSO': '! ALERTA SOSPECHOSA !'
            }
            cv2.putText(frame, alert_texts.get(risk_level, ''), (w//4 + 20, h-alert_h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        
        # Zonas de referencia
        y_range = self.config['zones']['steering_wheel']['y_range']
        x_range = self.config['zones']['steering_wheel']['x_range']
        
        steering_y_min = int(h * y_range[0])
        steering_y_max = int(h * y_range[1])
        steering_x_min = int(w * x_range[0])
        steering_x_max = int(w * x_range[1])
        
        cv2.rectangle(frame, (steering_x_min, steering_y_min), 
                     (steering_x_max, steering_y_max), (100, 100, 100), 2)
        cv2.putText(frame, 'ZONA VOLANTE', (steering_x_min + 10, steering_y_min + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return frame
    
    def process_frame(self, frame):
        start_time = time.time()
        
        self.video_recorder.add_frame(frame)
        
        yolo_results = self.yolo_model(frame, verbose=False, conf=self.config['models']['detection_conf'])
        pose_results = self.pose_model(frame, verbose=False, conf=self.config['models']['pose_conf'])
        
        keypoints_analysis = None
        if pose_results and len(pose_results[0].keypoints) > 0:
            keypoints_analysis = self.analyze_keypoints(pose_results[0].keypoints.data)
        
        # Detectar personas
        person_count = sum(1 for r in yolo_results for box in r.boxes 
                          if r.names[int(box.cls[0])] == 'person' and float(box.conf[0]) > 0.5)
        
        # Análisis
        hands_info = self.detect_hands_on_steering_wheel(keypoints_analysis, frame.shape)
        gaze_info = self.detect_gaze_direction(keypoints_analysis, frame.shape)
        objects = self.detect_suspicious_objects(yolo_results, hands_info)
        
        # Evaluar violaciones
        violations = self.evaluate_violations(hands_info, gaze_info, objects, person_count)
        
        risk_level, confidence, description = self.calculate_risk_score(violations)
        
        alert = self.generate_alert(violations, risk_level)
        if alert:
            print(f"\n[ALERTA {alert['risk_level']}]: {description}")
            if violations:
                for v in violations[:2]:
                    print(f"   - {v['description']} (duracion: {v['duration']:.1f}s)")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar en carpeta según nivel de riesgo
            if alert['risk_level'] == 'CRITICO':
                clip_path = f"clips/criticas/violation_{timestamp}.mp4"
            elif alert['risk_level'] == 'ALTO_RIESGO':
                clip_path = f"clips/alto_riesgo/violation_{timestamp}.mp4"
            else:  # SOSPECHOSO
                clip_path = f"clips/sospechosas/violation_{timestamp}.mp4"
            
            objects_str = ", ".join([o['object'] for o in objects]) if objects else ""
            
            if self.video_recorder.save_clip(clip_path):
                self.save_violation(alert, clip_path, objects_str)
            else:
                self.save_violation(alert, None, objects_str)
        
        # Dibujar anotaciones
        annotated_frame = yolo_results[0].plot() if not objects else frame.copy()
        if pose_results and len(pose_results[0].keypoints) > 0:
            annotated_frame = pose_results[0].plot()
        
        annotated_frame = self.draw_advanced_info(
            annotated_frame, hands_info, gaze_info, objects,
            violations, risk_level, confidence, description, person_count
        )
        
        self.fps = 1.0 / (time.time() - start_time)
        self.frame_count += 1
        
        return annotated_frame, alert
    
    def run(self, source=0, mode='auto', max_frames=9999):
        print("\n" + "="*70)
        print("SISTEMA AVANZADO DE MONITOREO DE CONDUCTORES")
        print("   Deteccion de Violaciones de Seguridad en Tiempo Real")
        print("="*70)
        print(f"Fuente: {source}")
        
        if mode == 'auto':
            try:
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_frame)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                mode = 'opencv'
                print("Modo: OpenCV (ventana nativa)")
            except:
                mode = 'headless'
                print("Modo: Headless (sin ventana, guardando frames)")
        
        print("\nCONTROLES:")
        if mode == 'opencv':
            print("   'q' - Salir del sistema")
            print("   's' - Capturar pantalla")
            print("   'r' - Ver reporte de sesion")
            print("   'd' - Toggle debug mode")
        print("="*70 + "\n")
        
        os.makedirs('output_frames', exist_ok=True)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la camara/video")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video_recording']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video_recording']['resolution'][1])
        cap.set(cv2.CAP_PROP_FPS, self.config['video_recording']['fps'])
        
        session_start = datetime.now()
        total_violations = {'CRITICO': 0, 'ALTO_RIESGO': 0, 'SOSPECHOSO': 0}
        
        try:
            frame_counter = 0
            while frame_counter < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Fin del video o error de captura")
                    break
                
                processed_frame, alert = self.process_frame(frame)
                
                if alert:
                    total_violations[alert['risk_level']] += 1
                
                if mode == 'opencv':
                    cv2.imshow('Sistema Avanzado de Monitoreo - ELO328', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Captura guardada: {filename}")
                    elif key == ord('r'):
                        self.print_session_report(session_start, total_violations)
                    elif key == ord('d'):
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                
                frame_counter += 1
        
        except KeyboardInterrupt:
            print("\nInterrupcion del usuario")
        
        finally:
            session_end = datetime.now()
            self.save_session_stats(session_start, session_end, total_violations)
            
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()
            
            print("\n" + "="*70)
            print("SESION FINALIZADA")
            print("="*70)
            self.print_session_report(session_start, total_violations)
            
            print(f"\nClips guardados en:")
            print(f"   - Criticas: clips/criticas/")
            print(f"   - Alto Riesgo: clips/alto_riesgo/")
            print(f"   - Sospechosas: clips/sospechosas/")
    
    def print_session_report(self, session_start, violations):
        duration = (datetime.now() - session_start).total_seconds()
        
        print("\nREPORTE DE SESION:")
        print("-" * 70)
        print(f"Duracion: {duration/60:.1f} minutos")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {self.fps:.1f}")
        print(f"\nVIOLACIONES DETECTADAS:")
        print(f"   - Criticas: {violations['CRITICO']}")
        print(f"   - Alto Riesgo: {violations['ALTO_RIESGO']}")
        print(f"   - Sospechosas: {violations['SOSPECHOSO']}")
        print(f"   - TOTAL: {sum(violations.values())}")
        print("-" * 70)
    
    def save_session_stats(self, start, end, violations):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO statistics (session_start, session_end, total_frames,
                                      total_violations, critical_count, high_risk_count,
                                      suspicious_count, avg_fps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                start.isoformat(),
                end.isoformat(),
                self.frame_count,
                sum(violations.values()),
                violations['CRITICO'],
                violations['ALTO_RIESGO'],
                violations['SOSPECHOSO'],
                self.fps
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Error guardando estadisticas: {e}")

if __name__ == "__main__":
    try:
        print("Inicializando Sistema Avanzado de Monitoreo...")
        system = AdvancedDriverMonitoring(config_path='config.yaml')
        
        SOURCE = 0
        MODE = 'auto'
        
        system.run(source=SOURCE, mode=MODE)
        
    except Exception as e:
        print(f"\nERROR CRITICO: {e}")
        import traceback
        traceback.print_exc()
