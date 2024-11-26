import cv2
import numpy as np

def calc():
    import detect
    import detect02
    while True:
        img = np.zeros((1080, 1920, 3), np.uint8)
        img.fill(255)
        # 마스터 카메라 정보
        cv2.putText(img, "Cam01 Area : " + str(100) + "m^2", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Cam01 People : " + str(detect.calcable), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Cam01 Density: " + str(round((detect.calcable / 7), 2)) + "%", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # 슬레이브 카메라1 정보
        cv2.putText(img, "Cam02 Area : " + str(100) + "m^2", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Cam02 People : " + str(detect.calcablee), (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Cam02 Density: " + str(round((detect.calcablee / 7), 2)) + "%", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 3)

        safenum = ((detect.calcable / 7) + (detect.calcablee / 7)) / 2
        if safenum >= 100:
            safestr = "Critical"
        elif safenum >= 85:
            safestr = "Danger"
        elif safenum >= 70:
            safestr = "Warning"
        else:
            safestr = "Safe"

        cv2.putText(img, "Area Safety : " + str(safestr), (20, 570), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        '''
        cv2.imshow("FRAME", img)
        if cv2.waitKey(1) == ord('q'):  # q를 눌러 나가기
            cv2.destroyAllWindows()
            break
        '''

        yield img


if __name__ == '__main__':
    calc()