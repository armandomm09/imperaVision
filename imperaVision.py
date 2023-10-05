import cv2
import mediapipe as mp
import math
from networktables import NetworkTables
import robotpy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializa la webcam
cap = cv2.VideoCapture(0)

# Inicializa MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    # Inicializa la conexión con NetworkTables
    NetworkTables.initialize(server='roborio-5887-frc.local')  

    table = NetworkTables.getTable("Vision")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convierte la imagen a formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa la imagen con MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Obtiene las coordenadas de los puntos clave de la mano
                points = []
                for point in landmarks.landmark:
                    x, y, _ = (point.x, point.y, point.z)
                    points.append((x, y))

                # Calcula el ángulo de giro de la mano (por ejemplo, el ángulo entre el dedo índice y el pulgar)
                angle = math.degrees(math.atan2(points[4][1] - points[8][1], points[4][0] - points[8][0]))

                # Normaliza el ángulo al rango [-1.00, 1.00]
                normalized_angle = (angle / 180.0) - 0.5

                # Dibuja el ángulo en la pantalla
                cv2.putText(frame, f'Drive Velocity: {normalized_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #Agrega el valor a Network Tables
                table.putNumber("Drive Velocity", normalized_angle)
                
                # Dibuja los puntos clave de la mano en la imagen
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Muestra el frame en la ventana
        cv2.imshow('Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera la captura y cierra la ventana
cap.release()
cv2.destroyAllWindows()
