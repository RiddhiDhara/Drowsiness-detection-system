import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import pyttsx3
import threading
import csv
from datetime import datetime
import time
import subprocess
from collections import deque

class DrowsinessDetectionSystem(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Drowsiness Detection System")
        self.geometry("900x700")
        self.configure(bg="#2c3e50")
    
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        # Main window
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#D2D0A0")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.header_frame = ctk.CTkFrame(self.main_frame, corner_radius=50, fg_color="#526E48")
        self.header_frame.pack(padx=10, pady=10, fill="x")

        self.header_label = ctk.CTkLabel(self.header_frame, text="Drowsiness Detection System", text_color="white", font=("Arial", 24))
        self.header_label.pack(pady=10)
        
        # Video window
        self.video_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#121212")
        self.video_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.video_label = ctk.CTkLabel(self.video_frame, text="", font=("Arial", 20), bg_color="#121212")
        self.video_label.pack(pady=10, fill="both", expand=True)
        
        # Status window
        self.status_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#27391C")
        self.status_frame.pack(pady=10, padx=50)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Awake\nNo signs detected", text_color="green", font=("Arial", 20, "bold"))
        self.status_label.pack(pady=10, padx=50)

        # Buttons
        self.button_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="transparent")
        self.button_frame.pack(pady=10, padx=10, fill="x")

        self.start_button = ctk.CTkButton(self.button_frame, text="Start Detection",fg_color="#A4B465", hover_color="#A5B68D",text_color="#2C3930",  font=("Arial", 16, "bold"),  height=50, width=200, corner_radius=20, command=self.start_detection)
        self.start_button.pack(pady=10, side="left", expand=True)

        self.stat_button = ctk.CTkButton(self.button_frame, text="Show Statistics",fg_color="#A4B465", hover_color="#A5B68D",text_color="#2C3930",  font=("Arial", 16, "bold"),  height=50, width=200, corner_radius=20, command=self.show_statistics)
        self.stat_button.pack(pady=10, side="left", expand=True)
        
        self.stop_button = ctk.CTkButton(self.button_frame, text="Stop Detection",fg_color="#A4B465", hover_color="#A5B68D",text_color="#2C3930",  font=("Arial", 16, "bold"),  height=50, width=200, corner_radius=20, command=self.stop_detection)
        self.stop_button.pack(pady=10, side="right", expand=True)
        
        # Detection parameters
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.cap = None
        self.is_detection_running = False
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
        self.MAR_THRESHOLD = 0.6    # Mouth Aspect Ratio threshold
        self.CLOSED_EYE_FRAMES = 30  # ~1 second at 30fps
        self.LONG_BLINK_THRESHOLD = 0.4  # Seconds
        self.HEAD_MOVEMENT_THRESHOLD = 0.2  # Threshold for extreme movements
        
        # Timing parameters
        self.WARNING_DURATION = {
            'single_sign': 5,      # 5 seconds for single sign
            'multiple_signs': 10,  # 10 seconds for multiple signs
            'drowsiness': 15       # 15 seconds for confirmed drowsiness
        }
        self.DROWSINESS_TIME_WINDOW = 600  # 10 minutes for drowsiness confirmation
        self.SIGN_COMBINATION_THRESHOLD = 3  # 3 combinations to confirm drowsiness
        
        # State tracking
        self.active_signs = set()
        self.warning_end_time = 0
        self.sign_history = deque(maxlen=20)
        self.drowsiness_combinations = 0
        self.last_sign_time = 0
        self.is_showing_warning = False
        self.current_warning = ""
        
        # Data tracking
        self.drowsiness_data = []

    def play_voice_alert(self, message):
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"Voice alert error: {e}")
    
    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[1] - mouth[5])
        B = np.linalg.norm(mouth[2] - mouth[4])
        C = np.linalg.norm(mouth[0] - mouth[3])
        return (A + B) / (2.0 * C)
    
    def detect_head_movement(self, landmarks):
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        
        current_position = np.array([
            (nose_tip.x + chin.x + left_ear.x + right_ear.x) / 4,
            (nose_tip.y + chin.y + left_ear.y + right_ear.y) / 4
        ])
        
        face_size = np.linalg.norm(np.array([left_ear.x, left_ear.y]) - 
                   np.array([right_ear.x, right_ear.y]))
        
        if not hasattr(self, 'prev_face_position'):
            self.prev_face_position = current_position
            return False
        
        movement = np.linalg.norm(current_position - self.prev_face_position) / face_size
        self.prev_face_position = current_position
        
        return movement > self.HEAD_MOVEMENT_THRESHOLD

    def update_warning_state(self):
        current_time = time.time()
        
        # Check if warning period has ended
        if self.is_showing_warning and current_time > self.warning_end_time:
            self.is_showing_warning = False
            self.active_signs.clear()
            self.status_label.configure(text_color="green", text="Status: Awake\nNo signs detected")
        
        # Check for drowsiness confirmation
        if (current_time - self.last_sign_time < self.DROWSINESS_TIME_WINDOW and 
            len(self.active_signs) >= 2 and 
            self.drowsiness_combinations >= self.SIGN_COMBINATION_THRESHOLD):
            
            if not self.is_showing_warning or "Drowsiness" not in self.current_warning:
                self.current_warning = f"Drowsiness detected: {', '.join(self.active_signs)}"
                self.status_label.configure(text_color="red", text=f"Status: Drowsy\n{self.current_warning}")
                threading.Thread(target=self.play_voice_alert, args=("Drowsiness detected! Please take a break",)).start()
                self.warning_end_time = current_time + self.WARNING_DURATION['drowsiness']
                self.is_showing_warning = True
                self.drowsiness_data.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Drowsiness', ', '.join(self.active_signs)])
        
        # Check for multiple signs warning
        elif len(self.active_signs) >= 2:
            if not self.is_showing_warning or "Early signs" not in self.current_warning:
                self.current_warning = f"Early signs detected: {', '.join(self.active_signs)}"
                self.status_label.configure(text_color="orange", text=f"Status: Warning\n{self.current_warning}")
                threading.Thread(target=self.play_voice_alert, args=("Early signs detected",)).start()
                self.warning_end_time = current_time + self.WARNING_DURATION['multiple_signs']
                self.is_showing_warning = True
                self.drowsiness_combinations += 1
                self.drowsiness_data.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Warning', ', '.join(self.active_signs)])
        
        # Check for single sign warning
        elif len(self.active_signs) == 1:
            if not self.is_showing_warning or list(self.active_signs)[0] not in self.current_warning:
                self.current_warning = f"Sign detected: {list(self.active_signs)[0]}"
                self.status_label.configure(text_color="orange", text=f"Status: Caution\n{self.current_warning}")
                threading.Thread(target=self.play_voice_alert, args=("Warning sign detected",)).start()
                self.warning_end_time = current_time + self.WARNING_DURATION['single_sign']
                self.is_showing_warning = True
                self.drowsiness_data.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Caution', list(self.active_signs)[0]])
    
    def detect_drowsiness(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.detect_drowsiness)
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        current_time = time.time()

        # Clear signs if no new signs detected for 10 seconds
        if current_time - self.last_sign_time > 10 and self.active_signs:
            self.active_signs.clear()
            if not self.is_showing_warning:
                self.status_label.configure(text_color="green", text="Status: Awake\nNo signs detected")

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

                # Eye landmarks and EAR calculation
                left_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]])
                right_eye = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]])
                left_EAR = self.eye_aspect_ratio(left_eye)
                right_EAR = self.eye_aspect_ratio(right_eye)
                EAR = (left_EAR + right_EAR) / 2.0
                
                # Prolonged blink detection
                if EAR < self.EAR_THRESHOLD:
                    if not hasattr(self, 'blink_start_time'):
                        self.blink_start_time = current_time
                    elif current_time - self.blink_start_time > self.LONG_BLINK_THRESHOLD:
                        self.active_signs.add("Prolonged blink")
                        self.last_sign_time = current_time
                        self.sign_history.append(("Prolonged blink", current_time))
                else:
                    if hasattr(self, 'blink_start_time'):
                        del self.blink_start_time

                # Yawning detection
                mouth = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [61, 291, 317, 402, 268, 14]])
                MAR = self.mouth_aspect_ratio(mouth)
                if MAR > self.MAR_THRESHOLD:
                    self.active_signs.add("Yawning")
                    self.last_sign_time = current_time
                    self.sign_history.append(("Yawning", current_time))

                # Head movement detection
                if self.detect_head_movement(face_landmarks):
                    self.active_signs.add("Head tilt")
                    self.last_sign_time = current_time
                    self.sign_history.append(("Head tilt", current_time))

                # Visual feedback
                left_eye_center = (int(left_eye[0][0] * frame.shape[1]), int(left_eye[0][1] * frame.shape[0]))
                right_eye_center = (int(right_eye[0][0] * frame.shape[1]), int(right_eye[0][1] * frame.shape[0]))
                eye_color = (0, 0, 255) if "Prolonged blink" in self.active_signs else (0, 255, 0)
                cv2.circle(frame, left_eye_center, 5, eye_color, -1)
                cv2.circle(frame, right_eye_center, 5, eye_color, -1)

        # Update warning state
        self.update_warning_state()

        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

        if self.is_detection_running:
            self.after(10, self.detect_drowsiness)
            
    def show_statistics(self):
        try:
            
            result = subprocess.check_output(["python", "./python/2.py"]).decode("utf-8")
            
            # with open('drowsiness_data.csv', 'w', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(['Time', 'Alert Level', 'Signs Detected'])
            #     writer.writerows(self.drowsiness_data)
            
            stat_window = ctk.CTkToplevel(self)
            stat_window.title("Drowsiness Statistics")
            stat_window.geometry("600x400")

            frame = ctk.CTkFrame(stat_window)
            frame.pack(fill="both", expand=True, padx=10, pady=10)

            text_widget = ctk.CTkTextbox(frame, wrap="word", width=550, height=350)
            text_widget.pack(side="left", fill="both", expand=True)

            scrollbar = ctk.CTkScrollbar(frame, command=text_widget.yview)
            scrollbar.pack(side="right", fill="y")

            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # stats_text = "Drowsiness Alerts:\n\n"
            # for event in self.drowsiness_data[-20:]:
            #     stats_text += f"{event[0]}: {event[1]} - {event[2]}\n"
            
            text_widget.insert("1.0", result)
            text_widget.configure(state="disabled")

        except Exception as e:
            print(f"Error showing statistics: {e}")

    def start_detection(self):
        self.status_label.configure(text_color="green", text="Status: Awake\nNo signs detected")
        
        # Reset all tracking variables
        self.active_signs = set()
        self.sign_history = deque(maxlen=20)
        self.drowsiness_combinations = 0
        self.last_sign_time = 0
        self.is_showing_warning = False
        self.current_warning = ""
        self.drowsiness_data = []
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text_color="red", text="Error: Camera not accessible")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 780)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
        self.is_detection_running = True
        self.detect_drowsiness()
        
    def stop_detection(self):
        self.is_detection_running = False
        if self.cap:
            self.cap.release()
            
        # Show black screen
        black_image = np.zeros((420, 780, 3), dtype=np.uint8)
        black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(black_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk
        self.status_label.configure(text_color="red", text="Detection Stopped")
        
        # Save data to CSV
        with open('./python/drowsiness_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Alert Level', 'Signs Detected'])
            writer.writerows(self.drowsiness_data)

if __name__ == "__main__":
    app = DrowsinessDetectionSystem()
    app.mainloop()
