import mediapipe as mp
import cv2
import math

class HumanPose():
    def __init__(self):
        # Khởi tạo lớp mediapipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Khởi tạo lớp mediapipe drawing
        self.mp_drawing = mp.solutions.drawing_utils

        # Lưu lại vị trí của đường kẻ ngang 2 vai khi vỗ tay bắt đầu game
        self.shoudler_line_y = 0
      
    # Hình thành pose detection trên người chơi ở ảnh đầu vào
    def detectPose(self, image):
        # Chuyển ảnh từ BGR sang RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lấy kết quả pose
        results = self.pose.process(imageRGB)

        # Nếu nhận diện được pose
        if results.pose_landmarks:
            # Vẽ các điểm và đường nối
            self.mp_drawing.draw_landmarks(image, landmark_list=results.pose_landmarks, connections=self.mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 225, 255), thickness=3, circle_radius=3),
                                           connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        return image, results

    # Kiểm tra xem hiện tại người chơi đang ở bên trái, bên phải hay ở giữa
    def checkLeftRightCenter(self, image, results):
        # Lấy kích thước ảnh đầu vào: chiều cao và chiều rộng
        image_height, image_width, _ = image.shape
        
        # Chia đôi chiều rộng ảnh
        image_mid_width = image_width // 2

        # Lấy tọa độ của vai trái và vai phải theo x
        leftShoulder_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
        rightShoulder_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)

        # Kiểm tra trạng thại hiện tại của người chơi
        if (leftShoulder_x < image_mid_width) and (rightShoulder_x < image_mid_width):
            LRC = "Left"
        elif (leftShoulder_x > image_mid_width) and (rightShoulder_x > image_mid_width):
            LRC = "Right"
        else:
            LRC = "Center"

        # Hiển thị trạng thái hiện tại của người chơi lên ảnh
        cv2.putText(image, LRC, (5, image_height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(image, (image_mid_width, 0), (image_mid_width, image_height), (255, 255, 255), 2)

        return image, LRC

    # Kiểm tra xem hiện tại người chơi đang nhảy, đứng hay cúi
    def checkJumpStandCrouch(self, image, results):
        # Lấy kích thước ảnh đầu vào: chiều cao và chiều rộng
        image_height, image_width, _ = image.shape

        # Lấy tọa độ của vai trái và vai phải theo y
        leftShoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
        rightShoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)

        # Tính trung bình 2 tọa độ vai trái và vai phải theo y
        averageShoulder_y = abs(leftShoulder_y + rightShoulder_y) // 2

        jump_threshold = 50
        crouch_threshold = 50

        # Kiểm tra trạng thại hiện tại của người chơi
        if (averageShoulder_y < self.shoudler_line_y - jump_threshold):
            JSC = "Jump"
        elif (averageShoulder_y > self.shoudler_line_y + crouch_threshold):
            JSC = "Crouch"
        else:
            JSC = "Stand"

        # Hiển thị trạng thái hiện tại của người chơi lên ảnh
        cv2.putText(image, JSC, (5, image_height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        cv2.line(image, (0, self.shoudler_line_y), (image_width, self.shoudler_line_y), (0, 255, 255), 2)

        return image, JSC

    # Kiểm tra vỗ tay
    def checkClap(self, image, results):
        # Lấy kích thước ảnh đầu vào: chiều cao và chiều rộng 
        image_height, image_width, _ = image.shape

        # Lấy tọa độ cổ tay trái theo x và y
        left_hand = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                     results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)

        # Lấy tọa độ cổ tay phải theo x và y
        right_hand = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width,
                      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)

        # Tính khoảng cách giữa 2 cổ tay trái và phải
        distance = int(math.hypot(left_hand[0] - right_hand[0], left_hand[1] - right_hand[1]))

        # Kiểm tra trạng thái vỗ tay
        clap_threshold = 100
        if distance < clap_threshold:
            CLAP = "Clapping"
        else:
            CLAP = "No clapping"

        # Hiện thỉ trạng thái vỗ tay lên ảnh
        cv2.putText(image, CLAP, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

        return image, CLAP
    
    # Lấy vị trí của đường kẻ ngang 2 vai
    def saveShoulderLine_y(self, image, results):
        # Lấy kích thước ảnh đầu vào: chiều cao và chiều rộng
        image_height, image_width, _ = image.shape

        # Lấy tọa độ của vai trái và vai phải theo y
        leftShoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
        rightShoulder_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)

        # Chia đôi 2 tọa độ để có được tọa độ của đường kẻ ngang 2 vai
        self.shoudler_line_y = abs(leftShoulder_y + rightShoulder_y) // 2
        return
