import cv2
import mediapipe as mp
import pandas as pd

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Khởi tạo độ phân giải
cap.set(3, 1280)
cap.set(4, 960)

# Khởi tạo lớp mediapipe pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Khởi tạo lớp mediapipe drawing
mpDrawing = mp.solutions.drawing_utils

# Danh sách lưu thông số khung xương của các frame
lm_list = []

# Lấy 600 ảnh đầu vào
number_frames = 600

# Ghi nhận thông số x, y, z, visibility của mỗi frame
def makeLandmarkTimestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Lặp cho đến khi đủ 600 ảnh đầu vào
while len(lm_list) <= number_frames:
    # Đọc ảnh từ camera
    ret, frame = cap.read()

    # Nếu đọc được ảnh
    if ret:
        # Lật lại ảnh
        frame = cv2.flip(frame, 1)

        # Chuyển ảnh từ BGR sang RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Lấy kết quả pose
        results = pose.process(frameRGB)

        # Nếu nhận diện được pose
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = makeLandmarkTimestep(results)
            lm_list.append(lm)

            # Vẽ các điểm và đường nối
            mpDrawing.draw_landmarks(frame, landmark_list=results.pose_landmarks, connections=mpPose.POSE_CONNECTIONS,
                                     landmark_drawing_spec=mpDrawing.DrawingSpec(color=(255, 225, 255), thickness=3, circle_radius=3),
                                     connection_drawing_spec=mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Hiển thị ảnh lên camera
        cv2.imshow("Image", frame)

    # Ấn 'Q' để thoát khỏi camera
    if cv2.waitKey(1) == ord('q'):
        break

# Ghi vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv("Waving.txt")

# Hủy bỏ camera
cap.release()
cv2.destroyAllWindows()