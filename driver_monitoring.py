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
    def __init__(self, window_size=2, threshold=0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}
    
    def update(self, fault_type, is_detected):
        if fault_type not in self.history:
            self.history[fault_type] = deque(maxlen=self.window_size)
        
        self.history[fault_type].append(1 if is_detected else 0)
    
    def is_valid(self, fault_type):
        if fault_type not in self.history:
            return False
        
        if len(self.history[fault_type]) < 1:
            return False
        
        detection_rate = sum(self.history[fault_type]) / len(self.history[fault_type])
        return detection_rate >= self.threshold
    
    def reset(self, fault_type):
        if fault_type in self.history:
            self.history[fault_type].clear()


class VideoClipRecorder:
    def __init__(self, clip_duration=5.0, fps=30, codec='mp4v', resolution=(1280, 720), format='.mp4'):
        self.clip_duration = clip_duration
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.format = format
        self.buffer_size = int(clip_duration * fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)
    
    def add_frame(self, frame):
        self.frame_buffer.append(frame.copy())
    
    def get_current_buffer(self):
        return list(self.frame_buffer)
    
    def save_clip(self, output_path, frames=None):
        if frames is None:
            frames = list(self.frame_buffer)
        
        if len(frames) == 0:
            print(f"Error: Buffer vacio, no se puede guardar {output_path}")
            return False
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Probar múltiples codecs en orden de preferencia
            codec_options = [
                ('avc1', 'H.264 (mejor compatibilidad)'),
                ('H264', 'H.264 alternativo'),
                ('mp4v', 'MPEG-4 (fallback)'),
                ('XVID', 'Xvid (fallback 2)')
            ]
            
            out = None
            successful_codec = None
            
            for codec, codec_name in codec_options:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, self.fps, self.resolution)
                    
                    if out.isOpened():
                        successful_codec = codec_name
                        break
                    else:
                        out.release()
                        out = None
                except:
                    if out:
                        out.release()
                    out = None
                    continue
            
            if not out or not out.isOpened():
                print(f"Error: No se pudo inicializar VideoWriter con ningun codec")
                return False
            
            frames_written = 0
            for frame in frames:
                if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                    frame = cv2.resize(frame, self.resolution)
                out.write(frame)
                frames_written += 1
            
            out.release()
            print(f"✓ Clip guardado: {output_path} ({frames_written} frames) - Codec: {successful_codec}")
            return True
            
        except Exception as e:
            print(f"Error guardando clip: {e}")
            if out:
                out.release()
            return False


class AdvancedDriverMonitoring:
    def __init__(self, config_path='config.yaml'):
        print("Inicializando sistema avanzado...")
        
        self.config = self.load_config(config_path)
        
        self.yolo_model = YOLO(self.config['models']['detection'])
        self.pose_model = YOLO(self.config['models']['pose'])
        
        self.temporal_filter = TemporalFilter(
            window_size=self.config['temporal_filter']['window_size'],
            threshold=self.config['temporal_filter']['validation_threshold']
        )
        
        self.video_recorder = VideoClipRecorder(
            clip_duration=self.config['video_recording']['clip_duration'],
            fps=self.config['video_recording']['fps'],
            codec=self.config['video_recording']['codec'],
            resolution=tuple(self.config['video_recording']['resolution']),
            format='.mp4'
        )
        
        self.behavior_history = deque(maxlen=90)
        self.hand_position_history = deque(maxlen=60)
        self.gaze_history = deque(maxlen=30)
        
        self.current_faults = {}
        self.fault_start_times = {}
        self.last_saved_clips = {}
        
        self.fault_locks = {}
        self.lock_duration = 5.0
        
        self.frame_count = 0
        self.fps = 0
        
        self.debug_mode = True
        
        self.init_database()
        
        os.makedirs('captures', exist_ok=True)
        os.makedirs('faults', exist_ok=True)
        os.makedirs('clips/criticas', exist_ok=True)
        os.makedirs('clips/alto_riesgo', exist_ok=True)
        os.makedirs('clips/sospechosas', exist_ok=True)
        
        print("Sistema avanzado inicializado\n")
        self.print_safety_standards()
    
    def load_config(self, config_path):
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
                'object_detection': 0.3,
                'object_near_hand': 120,
                'gaze_deviation_sospechoso': 30,
                'gaze_deviation_alto_riesgo': 50
            },
            'temporal_filter': {
                'window_size': 2,
                'validation_threshold': 0.5
            },
            'video_recording': {
                'clip_duration': 5.0,
                'fps': 30,
                'codec': 'H264',
                'resolution': [1280, 720]
            },
            'safety_standards': {
                'CRITICO': {
                    'ambas_manos_fuera_volante': {'weight': 1.0, 'duration': 0.2},
                    'conductor_ausente': {'weight': 1.0, 'duration': 0.2},
                    'multiples_personas': {'weight': 0.9, 'duration': 0.2}
                },
                'ALTO_RIESGO': {
                    'una_mano_fuera_volante': {'weight': 0.75, 'duration': 0.2},
                    'uso_celular': {'weight': 0.85, 'duration': 0.2},
                    'mirada_muy_desviada': {'weight': 0.80, 'duration': 0.2}
                },
                'SOSPECHOSO': {
                    'objeto_detectado': {'weight': 0.60, 'duration': 0.2},
                    'mirada_desviada': {'weight': 0.55, 'duration': 0.2},
                    'manos_cerca_rostro': {'weight': 0.50, 'duration': 0.2}
                }
            }
        }
    
    def print_safety_standards(self):
        print("ESTANDARES DE CONDUCCION SEGURA:")
        print("="*70)
        for level, faults in self.config['safety_standards'].items():
            folder = level.lower().replace('_', '')
            print(f"\n[{level}] → clips/{folder}/")
            for fault, params in faults.items():
                print(f"   • {fault.replace('_', ' ').title()}")
                print(f"     Duracion: {params['duration']}s | Lock: {self.lock_duration}s")
        print("\n" + "="*70 + "\n")
    
    def init_database(self):
        self.conn = sqlite3.connect('driver_monitoring_advanced.db')
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                risk_level TEXT,
                fault_type TEXT,
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
                total_faults INTEGER,
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
        
        eye_center = [(left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2]
        
        center_x = w / 2
        nose_deviation = abs(nose[0] - center_x)
        max_deviation = w * 0.20
        
        deviation_percentage = (nose_deviation / max_deviation) * 100 if max_deviation > 0 else 0
        
        if left_ear[0] > 0 and right_ear[0] > 0:
            ear_distance = abs(left_ear[0] - right_ear[0])
            if ear_distance < w * 0.08:
                deviation_percentage += 20
        
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
    
    def detect_hands_near_face(self, keypoints_analysis):
        if not keypoints_analysis:
            return False
        
        nose = keypoints_analysis['nose']
        left_wrist = keypoints_analysis['left_wrist']
        right_wrist = keypoints_analysis['right_wrist']
        
        if nose[0] == 0:
            return False
        
        threshold = self.config['thresholds'].get('hand_face_distance', 150)
        
        left_near = False
        right_near = False
        
        if left_wrist[0] > 0:
            dist_left = np.sqrt((left_wrist[0] - nose[0])**2 + (left_wrist[1] - nose[1])**2)
            left_near = dist_left < threshold
        
        if right_wrist[0] > 0:
            dist_right = np.sqrt((right_wrist[0] - nose[0])**2 + (right_wrist[1] - nose[1])**2)
            right_near = dist_right < threshold
        
        return left_near or right_near
    
    def detect_suspicious_objects(self, yolo_results, hands_info):
        detected = []
        
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
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
    
    def evaluate_faults(self, hands_info, gaze_info, objects, person_count, keypoints_analysis):
        faults = []
        current_time = time.time()
        raw_faults = []
        
        if person_count == 0:
            raw_faults.append('conductor_ausente')
        
        if person_count > 1:
            raw_faults.append('multiples_personas')
        
        if hands_info['count'] == 0:
            raw_faults.append('ambas_manos_fuera_volante')
        
        if hands_info['count'] == 1:
            raw_faults.append('una_mano_fuera_volante')
        
        has_cellphone = False
        for obj in objects:
            if obj['object'] == 'cell phone':
                raw_faults.append('uso_celular')
                has_cellphone = True
                break
        
        if not has_cellphone and len(objects) > 0:
            raw_faults.append('objeto_detectado')
        
        if gaze_info['severity'] == 'SOSPECHOSO':
            raw_faults.append('mirada_desviada')
        elif gaze_info['severity'] == 'ALTO_RIESGO':
            raw_faults.append('mirada_muy_desviada')
        
        if self.detect_hands_near_face(keypoints_analysis):
            raw_faults.append('manos_cerca_rostro')
        
        if self.debug_mode and raw_faults and self.frame_count % 30 == 0:
            print(f"[DEBUG] Raw faults: {raw_faults}")
            print(f"[DEBUG] Locks activos: {list(self.fault_locks.keys())}")
        
        all_fault_types = set()
        for level_faults in self.config['safety_standards'].values():
            all_fault_types.update(level_faults.keys())
        
        for f_type in all_fault_types:
            is_detected = f_type in raw_faults
            self.temporal_filter.update(f_type, is_detected)
        
        for f_type in raw_faults:
            is_valid = self.temporal_filter.is_valid(f_type)
            
            if self.debug_mode and self.frame_count % 30 == 0:
                history = self.temporal_filter.history.get(f_type, [])
                rate = sum(history) / len(history) if history else 0
                print(f"[DEBUG] {f_type}: valid={is_valid}, rate={rate:.0%}")
            
            if not is_valid:
                continue
            
            level = None
            params = None
            for lvl, faults_dict in self.config['safety_standards'].items():
                if f_type in faults_dict:
                    level = lvl
                    params = faults_dict[f_type]
                    break
            
            if not level or not params:
                continue
            
            if f_type not in self.fault_start_times:
                self.fault_start_times[f_type] = current_time
            
            duration = current_time - self.fault_start_times[f_type]
            
            if duration < params['duration']:
                continue
            
            if f_type in self.fault_locks:
                time_since_last = current_time - self.fault_locks[f_type]
                if time_since_last < self.lock_duration:
                    if self.debug_mode and self.frame_count % 30 == 0:
                        print(f"[DEBUG] {f_type}: BLOQUEADO ({self.lock_duration - time_since_last:.1f}s restantes)")
                    continue
            
            description = self.get_fault_description(
                f_type, hands_info, gaze_info, objects, person_count
            )
            
            faults.append({
                'type': f_type,
                'level': level,
                'description': description,
                'confidence': 0.90,
                'weight': params['weight'],
                'duration': duration
            })
        
        current_types = set(raw_faults)
        for f_type in list(self.fault_start_times.keys()):
            if f_type not in current_types:
                del self.fault_start_times[f_type]
                self.temporal_filter.reset(f_type)
        
        return faults
    
    def get_fault_description(self, f_type, hands_info, gaze_info, objects, person_count):
        descriptions = {
            'ambas_manos_fuera_volante': 'AMBAS MANOS FUERA DEL VOLANTE',
            'una_mano_fuera_volante': 'Solo una mano en el volante',
            'conductor_ausente': 'CONDUCTOR NO DETECTADO',
            'multiples_personas': f'{person_count} personas en cabina',
            'uso_celular': 'USO DE CELULAR DETECTADO',
            'mirada_desviada': f'Mirada desviada ({gaze_info["deviation_angle"]:.0f}%)',
            'mirada_muy_desviada': f'Mirada muy desviada ({gaze_info["deviation_angle"]:.0f}%)',
            'objeto_detectado': f'{objects[0]["object"] if objects else "Objeto"} detectado',
            'manos_cerca_rostro': 'Manos muy cerca del rostro'
        }
        return descriptions.get(f_type, f_type.replace('_', ' ').title())
    
    def get_highest_risk_level(self, faults):
        if not faults:
            return 'NORMAL'
        
        priority = {'CRITICO': 3, 'ALTO_RIESGO': 2, 'SOSPECHOSO': 1, 'NORMAL': 0}
        highest = max(faults, key=lambda f: priority.get(f['level'], 0))
        return highest['level']
    
    def calculate_risk_score(self, faults):
        if not faults:
            return 'NORMAL', 0.95, 'Conduccion segura'
        
        primary_fault = max(faults, key=lambda f: f['weight'])
        
        level = primary_fault['level']
        confidence = primary_fault['confidence']
        description = primary_fault['description']
        
        if len(faults) > 1:
            description += f" (+{len(faults)-1} mas)"
        
        return level, confidence, description
    
    def save_fault(self, alert, hands_count=0, person_count=0, video_path=None, objects_detected=""):
        try:
            primary = alert['primary_fault']
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO faults (timestamp, risk_level, fault_type, 
                                      duration, confidence, hands_on_wheel, person_count,
                                      image_path, video_clip_path, objects_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['timestamp'],
                alert['risk_level'],
                primary['type'],
                primary['duration'],
                primary['confidence'],
                hands_count,
                person_count,
                f"faults/{alert['timestamp'].replace(':', '-')}.jpg",
                video_path,
                objects_detected
            ))
            self.conn.commit()
            print(f"✓ BD actualizada: {primary['type']}")
        except Exception as e:
            print(f"Error guardando en BD: {e}")
    
    def draw_advanced_info(self, frame, hands_info, gaze_info, objects, 
                          faults, risk_level, confidence, description, person_count):
        h, w = frame.shape[:2]
        
        colors = {
            'NORMAL': (0, 255, 0),
            'SOSPECHOSO': (0, 165, 255),
            'ALTO_RIESGO': (0, 140, 255),
            'CRITICO': (0, 0, 255)
        }
        color = colors.get(risk_level, (255, 255, 255))
        
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
        
        if faults:
            cv2.putText(frame, f'FALTAS: {len(faults)}',
                       (hand_panel_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            y = 125
            for i, f in enumerate(faults[:3]):
                cv2.putText(frame, f"{i+1}. {f['type'][:18]}",
                           (hand_panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)
                y += 20
        
        for obj in objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            obj_color = colors.get(obj['risk_level'], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 2)
            
            label = f"{obj['object']} {obj['confidence']:.0%}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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
        
        return frame
    
    def process_frame(self, frame):
        start_time = time.time()
        
        self.video_recorder.add_frame(frame)
        
        yolo_results = self.yolo_model(frame, verbose=False, conf=self.config['models']['detection_conf'])
        pose_results = self.pose_model(frame, verbose=False, conf=self.config['models']['pose_conf'])
        
        keypoints_analysis = None
        if pose_results and len(pose_results[0].keypoints) > 0:
            keypoints_analysis = self.analyze_keypoints(pose_results[0].keypoints.data)
        
        person_count = sum(1 for r in yolo_results for box in r.boxes 
                          if r.names[int(box.cls[0])] == 'person' and float(box.conf[0]) > 0.5)
        
        hands_info = self.detect_hands_on_steering_wheel(keypoints_analysis, frame.shape)
        gaze_info = self.detect_gaze_direction(keypoints_analysis, frame.shape)
        objects = self.detect_suspicious_objects(yolo_results, hands_info)
        
        faults = self.evaluate_faults(hands_info, gaze_info, objects, person_count, keypoints_analysis)
        
        highest_level = self.get_highest_risk_level(faults)
        risk_level, confidence, description = self.calculate_risk_score(faults)
        
        alerts_generated = []
        for fault in faults:
            fault_type = fault['type']
            fault_level = fault['level']
            
            alert = {
                'timestamp': datetime.now().isoformat(),
                'risk_level': fault_level,
                'faults': [fault],
                'primary_fault': fault,
                'total_faults': 1
            }
            
            print(f"\n{'='*60}")
            print(f"[{alert['risk_level']}] {fault['description']}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            folder = fault_level.lower().replace('_', '')
            clip_path = f"clips/{folder}/fault_{timestamp}_{fault_type}.mp4"
            
            print(f"→ Guardando: {clip_path}")
            
            current_buffer = self.video_recorder.get_current_buffer()
            objects_str = ", ".join([o['object'] for o in objects]) if objects else ""
            
            if self.video_recorder.save_clip(clip_path, frames=current_buffer):
                self.save_fault(alert, hands_info['count'], person_count, clip_path, objects_str)
                
                self.fault_locks[fault_type] = time.time()
                
                if fault_type in self.fault_start_times:
                    del self.fault_start_times[fault_type]
                self.temporal_filter.reset(fault_type)
                
                print(f"✓ Grabado | Lock: {self.lock_duration}s")
                alerts_generated.append(alert)
            else:
                self.save_fault(alert, hands_info['count'], person_count, None, objects_str)
            
            print(f"{'='*60}\n")
        
        annotated_frame = yolo_results[0].plot() if not objects else frame.copy()
        if pose_results and len(pose_results[0].keypoints) > 0:
            annotated_frame = pose_results[0].plot()
        
        annotated_frame = self.draw_advanced_info(
            annotated_frame, hands_info, gaze_info, objects,
            faults, highest_level, confidence, description, person_count
        )
        
        self.fps = 1.0 / (time.time() - start_time)
        self.frame_count += 1
        
        return annotated_frame, alerts_generated
    
    def run(self, source=0, mode='auto', max_frames=9999):
        print("\n" + "="*70)
        print("SISTEMA AVANZADO DE MONITOREO DE CONDUCTORES")
        print("="*70)
        print(f"Fuente: {source}")
        
        if mode == 'auto':
            try:
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_frame)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                mode = 'opencv'
                print("Modo: OpenCV")
            except:
                mode = 'headless'
                print("Modo: Headless")
        
        print("\nCONTROLES:")
        if mode == 'opencv':
            print("   'q' - Salir")
            print("   's' - Captura")
            print("   'd' - Debug ON/OFF")
        print("="*70 + "\n")
        
        os.makedirs('output_frames', exist_ok=True)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la fuente")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video_recording']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video_recording']['resolution'][1])
        cap.set(cv2.CAP_PROP_FPS, self.config['video_recording']['fps'])
        
        session_start = datetime.now()
        total_faults = {'CRITICO': 0, 'ALTO_RIESGO': 0, 'SOSPECHOSO': 0}
        
        try:
            frame_counter = 0
            while frame_counter < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, alerts = self.process_frame(frame)
                
                if alerts:
                    for alert in alerts:
                        total_faults[alert['risk_level']] += 1
                
                if mode == 'opencv':
                    cv2.imshow('Sistema Avanzado', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"Captura: {filename}")
                    elif key == ord('d'):
                        self.debug_mode = not self.debug_mode
                        print(f"Debug: {'ON' if self.debug_mode else 'OFF'}")
                
                frame_counter += 1
        
        except KeyboardInterrupt:
            print("\nInterrupcion del usuario")
        
        finally:
            session_end = datetime.now()
            self.save_session_stats(session_start, session_end, total_faults)
            
            cap.release()
            cv2.destroyAllWindows()
            self.conn.close()
            
            print("\n" + "="*70)
            print("SESION FINALIZADA")
            print("="*70)
            self.print_session_report(session_start, total_faults)
    
    def print_session_report(self, session_start, faults):
        duration = (datetime.now() - session_start).total_seconds()
        
        print(f"\nDuracion: {duration/60:.1f} min")
        print(f"Frames: {self.frame_count}")
        print(f"FPS: {self.fps:.1f}")
        print(f"\nFALTAS DETECTADAS:")
        print(f"   • Criticas: {faults['CRITICO']}")
        print(f"   • Alto Riesgo: {faults['ALTO_RIESGO']}")
        print(f"   • Sospechosas: {faults['SOSPECHOSO']}")
        print(f"   • TOTAL: {sum(faults.values())}")
    
    def save_session_stats(self, start, end, faults):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO statistics (session_start, session_end, total_frames,
                                      total_faults, critical_count, high_risk_count,
                                      suspicious_count, avg_fps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                start.isoformat(),
                end.isoformat(),
                self.frame_count,
                sum(faults.values()),
                faults['CRITICO'],
                faults['ALTO_RIESGO'],
                faults['SOSPECHOSO'],
                self.fps
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Error guardando estadisticas: {e}")


if __name__ == "__main__":
    try:
        print("Inicializando...")
        system = AdvancedDriverMonitoring(config_path='config.yaml')
        
        SOURCE = 0
        MODE = 'auto'
        
        system.run(source=SOURCE, mode=MODE)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
