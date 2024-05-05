import cv2
import pyautogui
import threading
import tensorflow as tf
import numpy as np
import time
from PoseDetection import HumanPose

class SubwaySurfers():
    def __init__(self):
        # Khởi tạo đối tượng pose
        self.pose = HumanPose()

        # Game chưa bắt đầu
        self.game_started = False

        '''
        0: Bên trái
        1: Ở giữa
        2: Bên phải
        '''
        self.x_position = 1

        '''
        0: Cúi
        1: Đứng
        2: Nhảy
        '''
        self.y_position = 1

        # Số frame mà người chơi vỗ tay
        self.clap_duration = 0

        # Danh sách lưu thông số khung xương của các frame
        self.lm_list = []

        # Cứ 10 frame nhận diện xem vẫy tay hay không
        self.number_timesteps = 10

        # Khởi tạo model
        self.model = tf.keras.models.load_model("model.h5")

        # Trạng thái Waving hay Standing
        self.label = "....."

        # Số frame mà người chơi vẫy tay
        self.waving_duration=0

        # Game chưa dừng
        self.pause = False

    # Thực hiện hành động: Di chuyển trái, phải
    def horizontalMovement(self, LRC):
        if LRC=="Left":
            for _ in range(self.x_position):
                pyautogui.press('left')
            self.x_position = 0
        elif LRC=="Right":
            for _ in range(2, self.x_position, -1):
                pyautogui.press('right')
            self.x_position = 2
        else:
            if self.x_position ==0:
                pyautogui.press('right')
            elif self.x_position == 2:
                pyautogui.press('left')
            self.x_position = 1
        return

    # Thực hiện hành động: Nhảy và cúi
    def verticalMovement(self, JSC):
        if (JSC=="Jump") and (self.y_position == 1):
            pyautogui.press('up')
            self.y_position = 2
        elif (JSC=="Crouch") and (self.y_position ==1):
            pyautogui.press('down')
            self.y_position = 0
        elif (JSC=="Stand") and (self.y_position !=1):
            self.y_position = 1
        return

    # Ghi nhận thông số x, y, z, visibility của mỗi frame
    def makeLandmarkTimestep(self, results):
        print(results.pose_landmarks.landmark)
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm
    
    # Kiểm tra vẫy tay
    def checkWaving(self):
        tmp = np.array(self.lm_list)
        tmp = np.expand_dims(self.lm_list, axis=0)
        print(tmp.shape)
        results = self.model.predict(tmp)
        print(results)
        if results[0][0] > 0.5:
            self.label = "Standing"
        else:
            self.label = "Waving"
        return

    # Dừng và tiếp tục trò chơi
    def pauseResume(self):
        if self.label == "Waving":
            self.waving_duration+=1
            if self.waving_duration == 20 and self.pause == True:
                pyautogui.click(x=1279, y=753, button="left")
                time.sleep(2)
                pyautogui.click(x=1098, y=913, button="left")
                self.pause = False
                self.waving_duration = 0
            elif self.waving_duration == 10 and self.pause == False:
                pyautogui.press('esc')
                self.pause = True
                self.waving_duration = 0
        else:
            self.waving_duration = 0

    # Chơi game
    def play(self):
        # Khởi tạo camera
        cap = cv2.VideoCapture(0)

        # Khởi tạo độ phân giải
        cap.set(3, 1280)
        cap.set(4, 960)

        while True:
            # Đọc ảnh từ camera
            ret, image = cap.read()
            
            # Nếu đọc được ảnh
            if ret:
                # Lật lại ảnh
                image = cv2.flip(image, 1)
                
                # Lấy kích thước ảnh đầu vào: chiều cao và chiều rộng
                image_height, image_width, _ = image.shape

                # Hình thành pose detection trên người chơi ở ảnh đầu vào
                image, results = self.pose.detectPose(image)

                # Nếu nhận diện được pose
                if results.pose_landmarks:
                    # Kiểm tra game đã bắt đầu chưa
                    if self.game_started:
                        # Kiểm tra trái phải
                        image, LRC = self.pose.checkLeftRightCenter(image, results)
                        self.horizontalMovement(LRC)

                        # Kiểm tra lên xuống
                        image, JSC = self.pose.checkJumpStandCrouch(image, results)
                        self.verticalMovement(JSC)

                        # Ghi nhận thông số khung xương
                        c_lm = self.makeLandmarkTimestep(results)
                        self.lm_list.append(c_lm)

                        # Đủ 10 frame thì nhận diện xem vẫy tay hay không và dừng trò chơi
                        if len(self.lm_list) == self.number_timesteps:
                            # Chạy nhận diện vẫy tay ở luồng riêng
                            thread1 = threading.Thread(target=self.checkWaving)
                            thread1.start()
                            self.lm_list = []

                        # Hiển thị trạng thái vẫy tay lên ảnh
                        cv2.putText(image, self.label, (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

                        # Dừng và tiếp tục trò chơi
                        thread2 = threading.Thread(target=self.pauseResume)
                        thread2.start()
                    else:
                        # Hiển thị thông báo lên ảnh
                        cv2.putText(image, "Clap your hand to start!", (5, image_height-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 3)

                    # Kiểm tra vỗ tay hay chưa
                    image, CLAP = self.pose.checkClap(image, results)
                    if CLAP == "Clapping":
                        self.clap_duration +=1

                        # Đủ 10 frame thì kiểm tra xem game đã bắt đầu chưa
                        if self.clap_duration == 10:
                            if self.game_started:
                                # Chết và chơi lại
                                self.x_position  = 1
                                self.y_position  = 1
                                self.pose.saveShoulderLine_y(image, results)
                                pyautogui.press('space')
                            else:
                                # Chơi mới
                                self.game_started  = True
                                self.pose.saveShoulderLine_y(image, results)
                                pyautogui.click(x=720, y = 560, button = "left")

                            self.clap_duration = 0
                    else:
                        self.clap_duration = 0

                # Hiển thị ảnh lên camera
                cv2.imshow("Virtual Camera of Game", image)

            # Ấn 'Q' để thoát khỏi camera
            if cv2.waitKey(1) == ord('q'):
                break

        # Hủy bỏ camera
        cap.release()
        cv2.destroyAllWindows()

# Khởi tạo đối tượng game
game = SubwaySurfers()

# Gọi phương thức play()
game.play()