# biometria_mejorado_pose.py
import cv2
import json
import os
import sys
import numpy as np
from datetime import datetime
try:
    import mediapipe as mp
except ImportError:
    print("Falta mediapipe. Instala con: pip install mediapipe")
    sys.exit(1)

SAVE_DIR = "faces"
THRESHOLD = 0.10
CAM_INDEX = 0
HAND_DIST_THRESHOLD = 0.15
AUTO_SAVE_COOLDOWN = 2.0
POSE_SAVE_STEP_DEG = 18.0

os.makedirs(SAVE_DIR, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

E_LEFT = 33
E_RIGHT = 263
NOSE = 1
MOUTH_L = 61
MOUTH_R = 291

def landmarks_to_embedding(landmarks):
    pts = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)
    if pts.shape[0] == 0:
        return None
    centroid = pts.mean(axis=0, keepdims=True)
    pts_centered = pts - centroid
    rms = np.sqrt((pts_centered ** 2).sum() / pts_centered.size)
    if rms < 1e-8:
        return None
    return (pts_centered / rms).flatten()

def emb_distance(a, b):
    if a is None or b is None or a.shape != b.shape:
        return np.inf
    return np.linalg.norm(a - b) / np.sqrt(a.size)

def get_head_angles(landmarks):
    L = np.array([landmarks[E_LEFT]["x"], landmarks[E_LEFT]["y"], landmarks[E_LEFT]["z"]], dtype=np.float32)
    R = np.array([landmarks[E_RIGHT]["x"], landmarks[E_RIGHT]["y"], landmarks[E_RIGHT]["z"]], dtype=np.float32)
    N = np.array([landmarks[NOSE]["x"], landmarks[NOSE]["y"], landmarks[NOSE]["z"]], dtype=np.float32)
    ML = np.array([landmarks[MOUTH_L]["x"], landmarks[MOUTH_L]["y"], landmarks[MOUTH_L]["z"]], dtype=np.float32)
    MR = np.array([landmarks[MOUTH_R]["x"], landmarks[MOUTH_R]["y"], landmarks[MOUTH_R]["z"]], dtype=np.float32)
    eye_vec = R - L
    roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))
    mid_eyes = (L + R) / 2.0
    face_forward = N - mid_eyes
    yaw = np.degrees(np.arctan2(face_forward[0], -face_forward[2] + 1e-8))
    vertical = ((ML + MR) / 2.0) - mid_eyes
    pitch = np.degrees(np.arctan2(-vertical[2], vertical[1] + 1e-8))
    return float(yaw), float(pitch), float(roll)

def load_database():
    db = []
    for fname in os.listdir(SAVE_DIR):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(SAVE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            emb = np.array(meta.get("embedding", []), dtype=np.float32)
            if emb.size == 0 and "landmarks" in meta:
                lms = meta["landmarks"]
                emb = landmarks_to_embedding(lms)
                if emb is not None:
                    meta["embedding"] = emb.tolist()
                    with open(path, "w", encoding="utf-8") as fw:
                        json.dump(meta, fw, ensure_ascii=False, indent=2)
            if emb is not None and emb.size > 0:
                db.append({
                    "name": meta.get("name", "desconocido"),
                    "image_path": meta.get("image_path", ""),
                    "embedding": emb,
                    "json_path": path,
                    "yaw": meta.get("yaw", None),
                    "pitch": meta.get("pitch", None),
                    "roll": meta.get("roll", None)
                })
        except:
            pass
    return db

def recognize(embedding, db):
    best_name, best_dist = None, np.inf
    for entry in db:
        d = emb_distance(embedding, entry["embedding"])
        if d < best_dist:
            best_dist = d
            best_name = entry["name"]
    if best_dist < THRESHOLD:
        return best_name, best_dist
    return None, np.inf

def save_sample(frame_bgr, bbox_xyxy, landmarks, name, yaw=None, pitch=None, roll=None, tag=None):
    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_img = frame_bgr[y1:y2, x1:x2].copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra = f"_{tag}" if tag else ""
    base = f"{ts}_{name.replace(' ', '_')}{extra}"
    img_path = os.path.join(SAVE_DIR, base + ".jpg")
    json_path = os.path.join(SAVE_DIR, base + ".json")
    cv2.imwrite(img_path, face_img)
    emb = landmarks_to_embedding(landmarks)
    meta = {
        "name": name,
        "datetime": datetime.now().isoformat(),
        "image_path": img_path,
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "landmarks": landmarks,
        "embedding": emb.tolist() if emb is not None else [],
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return img_path, json_path

def hand_near_face(hand_landmarks, face_landmarks):
    if not hand_landmarks or not face_landmarks:
        return False
    hand_pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    face_pts = np.array([[lm["x"], lm["y"]] for lm in face_landmarks])
    hand_center = hand_pts.mean(axis=0)
    face_center = face_pts.mean(axis=0)
    dist = np.linalg.norm(hand_center - face_center)
    return dist < HAND_DIST_THRESHOLD

def pose_far_from_existing(db, name, yaw, pitch, step_deg=POSE_SAVE_STEP_DEG):
    ys = []
    ps = []
    for e in db:
        if e["name"] == name and e["yaw"] is not None and e["pitch"] is not None:
            ys.append(e["yaw"]); ps.append(e["pitch"])
    if not ys:
        return True
    ys = np.array(ys); ps = np.array(ps)
    d = np.min(np.sqrt((ys - yaw) ** 2 + (ps - pitch) ** 2))
    return d >= step_deg

# ------------------- NUEVO: Registro guiado tipo Face ID -------------------
def registro_guiado(cap, face_det, face_mesh, name):
    instrucciones = [
        "Mira al frente",
        "Gira la cabeza a la izquierda",
        "Gira la cabeza a la derecha",
        "Inclina la cabeza hacia arriba",
        "Inclina la cabeza hacia abajo"
    ]
    db = load_database()
    for paso, instruccion in enumerate(instrucciones, start=1):
        print(f"\nPaso {paso}/{len(instrucciones)}: {instruccion}")
        start_time = datetime.now().timestamp()
        capturado = False
        while not capturado:
            ok, frame = cap.read()
            if not ok:
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]

            det_res = face_det.process(frame_rgb)
            bbox_xyxy = None
            if det_res.detections:
                d = det_res.detections[0]
                box = d.location_data.relative_bounding_box
                x1, y1 = int(box.xmin * W), int(box.ymin * H)
                x2, y2 = int((box.xmin + box.width) * W), int((box.ymin + box.height) * H)
                bbox_xyxy = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mesh_res = face_mesh.process(frame_rgb)
            landmarks = []
            yaw = pitch = roll = None
            if mesh_res.multi_face_landmarks:
                fml = mesh_res.multi_face_landmarks[0]
                for lm in fml.landmark:
                    landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
                if len(landmarks) > max(E_LEFT, E_RIGHT, NOSE, MOUTH_L, MOUTH_R):
                    yaw, pitch, roll = get_head_angles(landmarks)

            # Mostrar instrucciones en pantalla
            cv2.putText(frame, f"Registro: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, instruccion, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            cv2.imshow("Biometria mejorada (MediaPipe)", frame)
            cv2.waitKey(1)

            if bbox_xyxy and landmarks:
                if datetime.now().timestamp() - start_time > 2.0:  # espera 2 seg
                    save_sample(frame, bbox_xyxy, landmarks, name,
                                yaw=yaw, pitch=pitch, roll=roll,
                                tag=f"registro{paso}")
                    db = load_database()
                    print(f"✓ Capturado: {instruccion}")
                    capturado = True

# ------------------- MAIN -------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No pude abrir la cámara. Prueba cambiando CAM_INDEX.")
        return

    cv2.namedWindow("Biometria mejorada (MediaPipe)", cv2.WINDOW_NORMAL)
    db = load_database()
    last_auto_save = 0.0

    face_det = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = frame.shape[:2]

            det_res = face_det.process(frame_rgb)
            bbox_xyxy = None
            if det_res.detections:
                d = det_res.detections[0]
                box = d.location_data.relative_bounding_box
                x1, y1 = int(box.xmin * W), int(box.ymin * H)
                x2, y2 = int((box.xmin + box.width) * W), int((box.ymin + box.height) * H)
                bbox_xyxy = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mesh_res = face_mesh.process(frame_rgb)
            landmarks = []
            yaw = pitch = roll = None
            if mesh_res.multi_face_landmarks:
                fml = mesh_res.multi_face_landmarks[0]
                for lm in fml.landmark:
                    landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
                if len(landmarks) > max(E_LEFT, E_RIGHT, NOSE, MOUTH_L, MOUTH_R):
                    yaw, pitch, roll = get_head_angles(landmarks)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=fml,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            hand_res = hands.process(frame_rgb)
            hand_landmarks = hand_res.multi_hand_landmarks[0] if hand_res.multi_hand_landmarks else None

            label = "desconocido"
            dist_txt = ""
            who = None
            if landmarks and len(db) > 0:
                emb = landmarks_to_embedding(landmarks)
                if emb is not None:
                    who, dist = recognize(emb, db)
                    if who is not None:
                        label = f"{who}"
                        dist_txt = f"{dist:.3f}"

            now_ts = datetime.now().timestamp()
            if bbox_xyxy and landmarks and hand_near_face(hand_landmarks, landmarks):
                if now_ts - last_auto_save > AUTO_SAVE_COOLDOWN:
                    emb = landmarks_to_embedding(landmarks)
                    rname = who if who is not None else "auto"
                    save_sample(frame, bbox_xyxy, landmarks, rname, yaw=yaw, pitch=pitch, roll=roll, tag="mano")
                    db = load_database()
                    last_auto_save = now_ts

            if who is not None and yaw is not None and pitch is not None and bbox_xyxy and landmarks:
                if now_ts - last_auto_save > AUTO_SAVE_COOLDOWN and pose_far_from_existing(db, who, yaw, pitch, POSE_SAVE_STEP_DEG):
                    save_sample(frame, bbox_xyxy, landmarks, who, yaw=yaw, pitch=pitch, roll=roll, tag="pose")
                    db = load_database()
                    last_auto_save = now_ts

            if bbox_xyxy:
                x1, y1, x2, y2 = bbox_xyxy
                color = (0, 255, 0) if who is not None else (0, 0, 255)
                cv2.putText(frame, f"{label}", (x1, max(0, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if dist_txt:
                    cv2.putText(frame, f"dist {dist_txt}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            info = [
                "Comandos:",
                "[g] guardar manual  |  [q] salir",
                f"Muestras DB: {len(db)}",
                f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}" if yaw is not None else "Yaw/Pitch/Roll: -",
                f"Mano detectada: {'Si' if hand_landmarks else 'No'}",
                f"Autosave cooldown: {max(0, AUTO_SAVE_COOLDOWN - (datetime.now().timestamp() - last_auto_save)):.1f}s"
            ]
            for i, text in enumerate(info):
                cv2.putText(frame, text, (10, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Biometria mejorada (MediaPipe)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('g'):
                if bbox_xyxy and landmarks:
                    emb = landmarks_to_embedding(landmarks)
                    if emb is not None:
                        who_m, dist_m = recognize(emb, db)
                        if who_m is not None:
                            print(f"Ya registrado como '{who_m}' (dist {dist_m:.4f}). Se guarda variante.")
                            save_sample(frame, bbox_xyxy, landmarks, who_m, yaw=yaw, pitch=pitch, roll=roll, tag="manual")
                            db = load_database()
                        else:
                            print("\nEscribe un nombre/alias para registro guiado y presiona Enter:")
                            try:
                                name = input().strip()
                            except EOFError:
                                name = "sin_nombre"
                            if not name:
                                name = "sin_nombre"
                            registro_guiado(cap, face_det, face_mesh, name)
                            db = load_database()
                else:
                    print("No hay rostro listo para guardar.")

            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
