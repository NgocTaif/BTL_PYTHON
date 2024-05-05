import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

# Trạng thái Waving hay Standing
label = "....."

# Cứ 10 frame nhận diện xem vẫy tay hay không
number_timesteps = 10

# Danh sách lưu thông số khung xương của các frame
lm_list = []

# Khởi tạo lớp mediapipe pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Khởi tạo lớp mediapipe drawing
mpDrawing = mp.solutions.drawing_utils

# Khởi tạo model
model = tf.keras.models.load_model("model.h5")

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Khởi tạo độ phân giải
cap.set(3, 1280)
cap.set(4, 960)

# Ghi nhận thông số x, y, z, visibility của mỗi frame
def makeLandmarkTimestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Kiểm tra vẫy tay
def checkWaving(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "Standing"
    else:
        label = "Waving"
    return label

# Sau 60 frame sau khi start bắt đầu nhận diện
i = 0
warmup_frames = 60

while True:
    # Đọc ảnh từ camera
    success, img = cap.read()

    # Chuyển ảnh từ BGR sang RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Lấy kết quả pose
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        # Nếu nhận diện được pose
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            c_lm = makeLandmarkTimestep(results)
            lm_list.append(c_lm)

            # Đủ 10 frame thì nhận diện xem vẫy tay hay không
            if len(lm_list) == number_timesteps:
                # Chạy nhận diện vẫy tay ở luồng riêng
                t1 = threading.Thread(target=checkWaving, args=(model, lm_list))
                t1.start()
                lm_list = []

            # Vẽ các điểm và đường nối
            mpDrawing.draw_landmarks(img, landmark_list=results.pose_landmarks, connections=mpPose.POSE_CONNECTIONS,
                                     landmark_drawing_spec=mpDrawing.DrawingSpec(color=(255, 225, 255), thickness=3, circle_radius=3),
                                     connection_drawing_spec=mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=2))

    # Hiển thị trạng thái vẫy tay lên ảnh
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Hiển thị ảnh lên camera
    cv2.imshow("Image", img)

    # Ấn 'Q' để thoát khỏi camera
    if cv2.waitKey(1) == ord('q'):
        break

# Hủy bỏ camera
cap.release()
cv2.destroyAllWindows()
