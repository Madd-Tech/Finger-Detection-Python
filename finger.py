
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices yang dipakai:
# thumb tip = 4
# index tip = 8, pip = 6
# middle tip = 12, pip = 10
# ring tip = 16, pip = 14
# pinky tip = 20, pip = 18

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]  # untuk ibu jari kita pakai index 3 (ip) sebagai pembanding

def count_fingers(hand_landmarks, hand_label, img_width, img_height):
    """
    Menghitung berapa jari yang terangkat pada satu tangan.
    hand_landmarks: satu hasil hand landmarks (NormalizedLandmarkList)
    hand_label: string 'Left' atau 'Right' dari MediaPipe (handedness)
    """
    lm = hand_landmarks.landmark
    fingers_up = 0

    # konversi koordinat ter-normalisasi ke pixel (jika perlu)
    # tapi perbandingan relatif (x atau y) cukup dilakukan pada nilai normal
    # Thumb: bandingkan x antara tip(4) dan ip(3) (atau mcp)
    # Catatan: kondisi bisa berlawanan tergantung mirror; kita flip frame jadi gunakan kondisi ini.
    thumb_tip_x = lm[4].x
    thumb_ip_x = lm[3].x

    # Jika hand_label == 'Right' : ibu jari berada di kiri gambar relatif ke jari lain saat terangkat
    # Banyak implementasi: jika thumb_tip_x < thumb_ip_x => thumb terbuka untuk tangan kanan (setelah flip frame)
    if hand_label == "Right":
        if thumb_tip_x < thumb_ip_x:
            fingers_up += 1
    else:  # Left hand
        if thumb_tip_x > thumb_ip_x:
            fingers_up += 1

    # Finger lainnya: jika tip.y < pip.y (y meningkat ke bawah), berarti terangkat
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for t, p in zip(tips, pips):
        if lm[t].y < lm[p].y:
            fingers_up += 1

    return fingers_up

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka webcam. Coba periksa koneksi kamera atau device index.")
        return

    # Gunakan MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        prev_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari webcam.")
                break

            # Mirror image (lebih natural untuk user)
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Proses deteksi tangan
            results = hands.process(img_rgb)

            total_fingers = 0

            if results.multi_hand_landmarks:
                # results.multi_handedness memberikan label per tangan (Left/Right)
                # Mereka berpasangan dalam urutan yang sama: gunakan zip
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' atau 'Right'
                    h, w, _ = frame.shape

                    # Hitung untuk tiap tangan
                    cnt = count_fingers(hand_landmarks, label, w, h)
                    total_fingers += cnt

                    # Gambar skeleton & landmark
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # Tampilkan label & count dekat per tangan (ambil koordinat landmark 0 - pergelangan)
                    wrist = hand_landmarks.landmark[0]
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    text = f"{label}: {cnt}"
                    cv2.putText(frame, text, (wx - 20, wy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Batasi maksimum 10 (meskipun logika seharusnya tidak melebihi 10)
            total_fingers = max(0, min(10, total_fingers))

            # Tampilkan total
            cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"Total Jari: {total_fingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # FPS
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time) if prev_time else 0
            prev_time = cur_time
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Finger Count (press 'q' to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
