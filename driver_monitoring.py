import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from datetime import datetime
import sqlite3
import os

class AdvancedDriverMonitoring:
    def __init__(self):
        print(" Inicializando sistema avanzado...")
        
        # Modelos YOLO
        self.yolo_model = YOLO('yolov8n.pt')
        self.pose_model = YOLO('yolov8n-pose.pt')
        
        # Buffers temporales
        self.pose_history = deque(maxlen=30)
        self.behavior_history = deque(maxlen=90)
        self.hand_position_history = deque(maxlen=60)
        
        # Estándares de conducción segura
        self.safety_standards = {
            'CRITICO': {
                'ambas_manos_fuera_volante': {'weight': 1.0, 'duration': 1.0},
                'conductor_ausente': {'weight': 1.0, 'duration': 0.5},
                'multiples_personas': {'weight': 0.9, 'duration': 2.0},
                'mirada_desviada_prolongada': {'weight': 0.85, 'duration': 2.0},
                'objeto_rostro_movimiento': {'weight': 0.95, 'duration': 1.0}
            },
            'ALTO_RIESGO': {
                'una_mano_fuera_volante_con_objeto': {'weight': 0.8, 'duration': 2.0},
                'postura_inadecuada': {'weight': 0.7, 'duration': 3.0},
                'uso_telefono': {'weight': 0.85, 'duration': 1.0},
                'bebiendo_comiendo': {'weight': 0.75, 'duration': 2.0}
            },
            'SOSPECHOSO': {
                'una_mano_cerca_rostro': {'weight': 0.6, 'duration': 2.0},
                'postura_reclinada': {'weight': 0.5, 'duration': 5.0},
                'objeto_sospechoso_visible': {'weight': 0.55, 'duration': 3.0},
                'mano_no_visible': {'weight': 0.5, 'duration': 4.0}
            }
        }
        
        # Configuración para cámara frontal
        self.config = {
            'steering_wheel_region_y': (0.5, 0.95),
            'steering_wheel_region_x': (0.2, 0.8),
            'face_region_y': (0.15, 0.55),
            'face_region_x': (0.3, 0.7),
            'hand_face_distance_threshold': 180,
            'hand_steering_distance_threshold': 250,
            'posture_angle_threshold': 20,
            'confidence_threshold': 0.5
        }
        
        # Estado del sistema
        self.current_violations = {}
        self.violation_start_times = {}
        self.alert_cooldown = {}
        self.frame_count = 0
        self.fps = 0
        
        # Base de datos
        self.init_database()
        os.makedirs('captures', exist_ok=True)
        os.makedirs('violations', exist_ok=True)
        
        print(" Sistema avanzado inicializado\n")
        self.print_safety_standards()
    
    def print_safety_standards(self):
        print(" ESTÁNDARES DE CONDUCCIÓN SEGURA:")
        print("="*70)
        for level, violations in self.safety_standards.items():
            print(f"\n🚨 {level}:")
            for violation, params in violations.items():
                print(f"   • {violation.replace('_', ' ').title()}")
                print(f"     Peso: {params['weight']:.0%} | Duración: {params['duration']}s")
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
                posture_angle REAL,
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
    
    def detect_hands_on_steering_wheel(self, keypoints_analysis, frame_shape):
        if not keypoints_analysis:
            return {'left': False, 'right': False, 'count': 0, 'left_pos': [0,0], 'right_pos': [0,0]}
        
        h, w = frame_shape[:2]
        
        steering_y_min = int(h * self.config['steering_wheel_region_y'][0])
        steering_y_max = int(h * self.config['steering_wheel_region_y'][1])
        steering_x_min = int(w * self.config['steering_wheel_region_x'][0])
        steering_x_max = int(w * self.config['steering_wheel_region_x'][1])
        
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
    
    def detect_hand_near_face(self, keypoints_analysis, frame_shape):
        if not keypoints_analysis:
            return {'left': False, 'right': False, 'distance_left': 999, 'distance_right': 999, 'closest': 'left', 'min_distance': 999}
        
        h, w = frame_shape[:2]
        
        nose = keypoints_analysis['nose']
        left_wrist = keypoints_analysis['left_wrist']
        right_wrist = keypoints_analysis['right_wrist']
        left_elbow = keypoints_analysis['left_elbow']
        right_elbow = keypoints_analysis['right_elbow']
        
        def calculate_distance(p1, p2):
            if p1[0] == 0 or p2[0] == 0:
                return 999
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        dist_left_wrist = calculate_distance(nose, left_wrist)
        dist_right_wrist = calculate_distance(nose, right_wrist)
        dist_left_elbow = calculate_distance(nose, left_elbow)
        dist_right_elbow = calculate_distance(nose, right_elbow)
        
        dist_left = min(dist_left_wrist, dist_left_elbow)
        dist_right = min(dist_right_wrist, dist_right_elbow)
        
        threshold = self.config['hand_face_distance_threshold']
        face_y_max = int(h * self.config['face_region_y'][1])
        
        left_near = dist_left < threshold and left_wrist[1] < face_y_max
        right_near = dist_right < threshold and right_wrist[1] < face_y_max
        
        return {
            'left': left_near,
            'right': right_near,
            'distance_left': dist_left,
            'distance_right': dist_right,
            'closest': 'left' if dist_left < dist_right else 'right',
            'min_distance': min(dist_left, dist_right)
        }
    
    def analyze_posture(self, keypoints_analysis):
        if not keypoints_analysis:
            return {'angle': 0, 'status': 'desconocido', 'is_proper': False}
        
        try:
            left_shoulder = keypoints_analysis['left_shoulder']
            right_shoulder = keypoints_analysis['right_shoulder']
            
            if left_shoulder[0] == 0 or right_shoulder[0] == 0:
                return {'angle': 0, 'status': 'desconocido', 'is_proper': False}
            
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            
            if angle < self.config['posture_angle_threshold']:
                status = 'correcta'
                is_proper = True
            elif angle < 45:
                status = 'inclinada'
                is_proper = False
            else:
                status = 'muy_inclinada'
                is_proper = False
            
            return {'angle': angle, 'status': status, 'is_proper': is_proper}
        except:
            return {'angle': 0, 'status': 'error', 'is_proper': False}
    
    def detect_suspicious_objects(self, yolo_results, hands_info):
        suspicious_classes = {
            'cell phone': 'CRITICO',
            'bottle': 'ALTO_RIESGO',
            'cup': 'ALTO_RIESGO',
            'wine glass': 'ALTO_RIESGO',
            'book': 'SOSPECHOSO',
            'remote': 'SOSPECHOSO'
        }
        
        detected = []
        
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in suspicious_classes and confidence > 0.4:
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    near_hand = False
                    if hands_info:
                        obj_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                        for hand_pos in [hands_info['left_pos'], hands_info['right_pos']]:
                            if hand_pos[0] > 0:
                                dist = np.sqrt((obj_center[0]-hand_pos[0])**2 + 
                                             (obj_center[1]-hand_pos[1])**2)
                                if dist < 100:
                                    near_hand = True
                                    break
                    
                    detected.append({
                        'object': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'risk_level': suspicious_classes[class_name],
                        'near_hand': near_hand
                    })
        
        return detected
    
    def evaluate_violations(self, hands_info, hand_face_info, posture_info, 
                          objects, person_count, face_detected):
        violations = []
        current_time = time.time()
        
        if hands_info['count'] == 0:
            violations.append({
                'type': 'ambas_manos_fuera_volante',
                'level': 'CRITICO',
                'description': '¡AMBAS MANOS FUERA DEL VOLANTE!',
                'confidence': 0.95,
                'weight': 1.0
            })
        
        if not face_detected or person_count == 0:
            violations.append({
                'type': 'conductor_ausente',
                'level': 'CRITICO',
                'description': 'Conductor no detectado en posición',
                'confidence': 0.90,
                'weight': 1.0
            })
        
        if person_count > 1:
            violations.append({
                'type': 'multiples_personas',
                'level': 'CRITICO',
                'description': f'{person_count} personas en cabina',
                'confidence': 0.85,
                'weight': 0.9
            })
        
        for obj in objects:
            if obj['near_hand'] and (hand_face_info['left'] or hand_face_info['right']):
                level = 'CRITICO' if obj['object'] == 'cell phone' else 'ALTO_RIESGO'
                violations.append({
                    'type': 'objeto_rostro_movimiento',
                    'level': level,
                    'description': f'Usando {obj["object"]} en rostro',
                    'confidence': obj['confidence'] * 0.95,
                    'weight': 0.95 if level == 'CRITICO' else 0.8
                })
        
        if hands_info['count'] == 1 and len(objects) > 0:
            violations.append({
                'type': 'una_mano_fuera_volante_con_objeto',
                'level': 'ALTO_RIESGO',
                'description': f'Una mano fuera + {objects[0]["object"]}',
                'confidence': 0.75,
                'weight': 0.8
            })
        
        if not posture_info['is_proper'] and posture_info['status'] != 'desconocido':
            violations.append({
                'type': 'postura_inadecuada',
                'level': 'ALTO_RIESGO',
                'description': f'Postura {posture_info["status"]} ({posture_info["angle"]:.1f}°)',
                'confidence': 0.70,
                'weight': 0.7
            })
        
        if hands_info['count'] == 1 and (hand_face_info['left'] or hand_face_info['right']):
            violations.append({
                'type': 'una_mano_cerca_rostro',
                'level': 'SOSPECHOSO',
                'description': f'Mano {hand_face_info["closest"]} cerca del rostro',
                'confidence': 0.65,
                'weight': 0.6
            })
        
        for obj in objects:
            if not obj['near_hand']:
                violations.append({
                    'type': 'objeto_sospechoso_visible',
                    'level': 'SOSPECHOSO',
                    'description': f'{obj["object"]} visible en cabina',
                    'confidence': obj['confidence'] * 0.6,
                    'weight': 0.55
                })
        
        validated_violations = []
        for violation in violations:
            v_type = violation['type']
            
            if v_type not in self.violation_start_times:
                self.violation_start_times[v_type] = current_time
            
            duration = current_time - self.violation_start_times[v_type]
            violation['duration'] = duration
            
            required_duration = self.safety_standards[violation['level']][v_type]['duration']
            if duration >= required_duration:
                validated_violations.append(violation)
        
        current_types = {v['type'] for v in violations}
        for v_type in list(self.violation_start_times.keys()):
            if v_type not in current_types:
                del self.violation_start_times[v_type]
        
        return validated_violations
    
    def calculate_risk_score(self, violations):
        if not violations:
            return 'NORMAL', 0.95, 'Conducción segura - Manos en volante'
        
        max_weight = max(v['weight'] for v in violations)
        primary_violation = max(violations, key=lambda v: v['weight'])
        
        level = primary_violation['level']
        confidence = primary_violation['confidence']
        description = primary_violation['description']
        
        if len(violations) > 1:
            description += f" (+{len(violations)-1} más)"
        
        return level, confidence, description
    
    def generate_alert(self, violations, risk_level):
        if not violations or risk_level == 'NORMAL':
            return None
        
        current_time = time.time()
        primary = max(violations, key=lambda v: v['weight'])
        
        cooldown = 2.0 if risk_level == 'CRITICO' else 5.0
        
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
        self.save_violation(alert)
        return alert
    
    def save_violation(self, alert):
        try:
            primary = alert['primary_violation']
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO violations (timestamp, risk_level, violation_type, 
                                      duration, confidence, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert['timestamp'],
                alert['risk_level'],
                primary['type'],
                primary['duration'],
                primary['confidence'],
                f"violations/{alert['timestamp'].replace(':', '-')}.jpg"
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Error guardando: {e}")
    
    def draw_advanced_info(self, frame, hands_info, hand_face_info, posture_info,
                          objects, violations, risk_level, confidence, description):
        h, w = frame.shape[:2]
        
        colors = {
            'NORMAL': (0, 255, 0),
            'SOSPECHOSO': (0, 165, 255),
            'ALTO_RIESGO': (0, 140, 255),
            'CRITICO': (0, 0, 255)
        }
        color = colors.get(risk_level, (255, 255, 255))
        
        panel_h = 220
        cv2.rectangle(frame, (10, 10), (w - 10, panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_h), color, 3)
        
        cv2.putText(frame, 'SISTEMA AVANZADO DE MONITOREO', (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f'ESTADO: {risk_level}', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        
        cv2.putText(frame, description[:60], (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        y_offset = 130
        cv2.putText(frame, f'Manos en volante: {hands_info["count"]}/2', (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if hands_info['count'] == 2 else (0, 0, 255), 1)
        
        y_offset += 25
        posture_color = (0, 255, 0) if posture_info['is_proper'] else (0, 165, 255)
        cv2.putText(frame, f'Postura: {posture_info["status"]} ({posture_info["angle"]:.1f}°)',
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 1)
        
        y_offset += 25
        cv2.putText(frame, f'Distancia mano-rostro: {hand_face_info["min_distance"]:.0f}px',
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f'FPS: {self.fps:.1f} | Confianza: {confidence:.0%}',
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
        
        if violations:
            cv2.putText(frame, f'VIOLACIONES ACTIVAS: {len(violations)}',
                       (hand_panel_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            y = 125
            for i, v in enumerate(violations[:3]):
                duration_text = f"{v['duration']:.1f}s"
                cv2.putText(frame, f"{i+1}. {v['type'][:20]} ({duration_text})",
                           (hand_panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)
                y += 20
        
        if risk_level == 'CRITICO':
            alert_h = 60
            cv2.rectangle(frame, (w//4, h-alert_h-20), (3*w//4, h-20), (0, 0, 255), -1)
            cv2.putText(frame, '!!! ALERTA CRITICA !!!', (w//4 + 20, h-alert_h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        
        steering_y_min = int(h * self.config['steering_wheel_region_y'][0])
        steering_y_max = int(h * self.config['steering_wheel_region_y'][1])
        steering_x_min = int(w * self.config['steering_wheel_region_x'][0])
        steering_x_max = int(w * self.config['steering_wheel_region_x'][1])
        
        cv2.rectangle(frame, (steering_x_min, steering_y_min), 
                     (steering_x_max, steering_y_max), (100, 100, 100), 2)
        cv2.putText(frame, 'ZONA VOLANTE', (steering_x_min + 10, steering_y_min + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        face_y_min = int(h * self.config['face_region_y'][0])
        face_y_max = int(h * self.config['face_region_y'][1])
        face_x_min = int(w * self.config['face_region_x'][0])
        face_x_max = int(w * self.config['face_region_x'][1])
        
        cv2.rectangle(frame, (face_x_min, face_y_min), 
                     (face_x_max, face_y_max), (100, 100, 100), 1)
        cv2.putText(frame, 'ZONA ROSTRO', (face_x_min + 10, face_y_min + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return frame
    
    def process_frame(self, frame):
        start_time = time.time()
        
        yolo_results = self.yolo_model(frame, verbose=False, conf=0.45)
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        
        keypoints_analysis = None
        if pose_results and len(pose_results[0].keypoints) > 0:
            keypoints_analysis = self.analyze_keypoints(pose_results[0].keypoints.data)
        
        hands_info = self.detect_hands_on_steering_wheel(keypoints_analysis, frame.shape)
        hand_face_info = self.detect_hand_near_face(keypoints_analysis, frame.shape)
        posture_info = self.analyze_posture(keypoints_analysis)
        
        objects = self.detect_suspicious_objects(yolo_results, hands_info)
        
        person_count = sum(1 for r in yolo_results for box in r.boxes 
                          if r.names[int(box.cls[0])] == 'person' and float(box.conf[0]) > 0.5)
        
        face_detected = person_count > 0
        
        violations = self.evaluate_violations(
            hands_info, hand_face_info, posture_info, 
            objects, person_count, face_detected
        )
        
        risk_level, confidence, description = self.calculate_risk_score(violations)
        
        alert = self.generate_alert(violations, risk_level)
        if alert:
            print(f"\n🚨 ALERTA [{alert['risk_level']}]: {description}")
            if violations:
                for v in violations[:2]:
                    print(f"   • {v['description']} (duración: {v['duration']:.1f}s)")
        
        annotated_frame = yolo_results[0].plot()
        if pose_results and len(pose_results[0].keypoints) > 0:
            annotated_frame = pose_results[0].plot()
        
        annotated_frame = self.draw_advanced_info(
            annotated_frame, hands_info, hand_face_info, posture_info,
            objects, violations, risk_level, confidence, description
        )
        
        self.fps = 1.0 / (time.time() - start_time)
        self.frame_count += 1
        
        return annotated_frame, alert
    
    def run(self, source=0, mode='auto', max_frames=9999):
        print("\n" + "="*70)
        print(" SISTEMA AVANZADO DE MONITOREO DE CONDUCTORES")
        print("   Detección de Violaciones de Seguridad en Tiempo Real")
        print("="*70)
        print(f" Fuente: {source}")
        
        # Detectar modo de visualización
        if mode == 'auto':
            try:
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_frame)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                mode = 'opencv'
                print(" Modo: OpenCV (ventana nativa)")
            except:
                try:
                    import matplotlib.pyplot as plt
                    mode = 'matplotlib'
                    print(" Modo: Matplotlib (ventana alternativa)")
                except:
                    mode = 'headless'
                    print("  Modo: Headless (sin ventana, guardando frames)")
        
        print("\n  CONTROLES:")
        if mode == 'opencv':
            print("   'q' - Salir del sistema")
            print("   's' - Capturar pantalla")
            print("   'r' - Ver reporte de sesión")
        elif mode == 'matplotlib':
            print("   Cerrar ventana - Salir del sistema")
            print("   Los frames se actualizan automáticamente")
        else:
            print("   Ctrl+C - Detener procesamiento")
            print("   Frames guardándose en /output_frames/")
        print("="*70 + "\n")
        
        os.makedirs('output_frames', exist_ok=True)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara/video")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        session_start = datetime.now()
        total_violations = {'CRITICO': 0, 'ALTO_RIESGO': 0, 'SOSPECHOSO': 0}
        
        if mode == 'matplotlib':
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 8))
            img_plot = None
        
        try:
            frame_counter = 0
            while frame_counter < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print(" Fin del video o error de captura")
                    break
                
                processed_frame, alert = self.process_frame(frame)
                
                if alert:
                    total_violations[alert['risk_level']] += 1
                
                if mode == 'opencv':
                    try:
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
                    except cv2.error as e:
                        print(f" Error OpenCV: {e}")
                        print("Cambiando a modo headless...")
                        mode = 'headless'
                
                elif mode == 'matplotlib':
                    try:
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        if img_plot is None:
                            img_plot = ax.imshow(frame_rgb)
                            ax.axis('off')
                            plt.title('Sistema de Monitoreo - ELO328', fontsize=14, fontweight='bold')
                        else:
                            img_plot.set_data(frame_rgb)
                        
                        plt.pause(0.001)
                        
                        if not plt.fignum_exists(fig.number):
                            print("Ventana cerrada por usuario")
                            break
                    except Exception as e:
                        print(f" Error Matplotlib: {e}")
                        mode = 'headless'
                
                elif mode == 'headless':
                    if frame_counter % 30 == 0:
                        filename = f"output_frames/frame_{frame_counter:04d}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        print(f"📸 Frame {frame_counter} | FPS: {self.fps:.1f} | Violaciones: {sum(total_violations.values())}")
                    
                    if alert:
                        filename = f"violations/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, processed_frame)
                
                frame_counter += 1
        
        except KeyboardInterrupt:
            print("\n Interrupción del usuario")
        
        finally:
            if mode == 'matplotlib':
                try:
                    plt.close('all')
                except:
                    pass
            
            session_end = datetime.now()
            self.save_session_stats(session_start, session_end, total_violations)
            
            cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
            self.conn.close()
            
            print("\n" + "="*70)
            print("SESIÓN FINALIZADA")
            print("="*70)
            self.print_session_report(session_start, total_violations)
            
            if mode == 'headless':
                print(f"\nframes guardados en: output_frames/")
                print(f"Alertas guardadas en: violations/")
    
    def print_session_report(self, session_start, violations):
        duration = (datetime.now() - session_start).total_seconds()
        
        print("\n📊 REPORTE DE SESIÓN:")
        print("-" * 70)
        print(f"⏱Duración: {duration/60:.1f} minutos")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {self.fps:.1f}")
        print(f"\nVIOLACIONES DETECTADAS:")
        print(f"   • Críticas: {violations['CRITICO']}")
        print(f"   • Alto Riesgo: {violations['ALTO_RIESGO']}")
        print(f"   • Sospechosas: {violations['SOSPECHOSO']}")
        print(f"   • TOTAL: {sum(violations.values())}")
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
            print(f"Error guardando estadísticas: {e}")


if __name__ == "__main__":
    try:
        print("🔧 Inicializando Sistema Avanzado de Monitoreo...")
        system = AdvancedDriverMonitoring()
        
        # CONFIGURACIÓN
        SOURCE = 0  # 0 = cámara web, 'video.mp4' = archivo
        
        # MODO: 'auto' detecta automáticamente el mejor método
        # 'opencv' = ventana nativa (más rápida)
        # 'matplotlib' = ventana alternativa (más compatible) 
        # 'headless' = sin ventana, guarda frames
        MODE = 'auto'
        
        system.run(source=SOURCE, mode=MODE)
        
    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()