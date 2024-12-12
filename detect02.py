from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from djitellopy import Tello

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, GridManager, GridConfig
from utils.torch_utils import select_device, load_classifier, time_synchronized

# 전역변수
slave_grid = np.zeros((5, 6), dtype=int)
temp_grid = np.zeros((5, 6), dtype=int)

# 주변 그리드 1차 보정 함수
def calculate_corrected_value(grid, i, j, w0, w1, w2):
    # 현재 셀의 값
    current_value = grid[i, j]

    # 상하좌우 인접 셀의 값
    udlr_values = 0
    if i > 0:  # 위쪽 셀
        udlr_values += grid[i - 1, j]
    if i < grid.shape[0] - 1:  # 아래쪽 셀
        udlr_values += grid[i + 1, j]
    if j > 0:  # 왼쪽 셀
        udlr_values += grid[i, j - 1]
    if j < grid.shape[1] - 1:  # 오른쪽 셀
        udlr_values += grid[i, j + 1]

    # 대각선 인접 셀들의 값
    diagonal_values = 0
    if i > 0 and j > 0:  # 왼쪽 위 대각선 셀
        diagonal_values += grid[i - 1, j - 1]
    if i > 0 and j < grid.shape[1] - 1:  # 오른쪽 위 대각선 셀
        diagonal_values += grid[i - 1, j + 1]
    if i < grid.shape[0] - 1 and j > 0:  # 왼쪽 아래 대각선 셀
        diagonal_values += grid[i + 1, j - 1]
    if i < grid.shape[0] - 1 and j < grid.shape[1] - 1:  # 오른쪽 아래 대각선 셀
        diagonal_values += grid[i + 1, j + 1]

    # 보정된 값 계산
    corrected_value = (w0 * current_value) + (w1 * udlr_values) + (w2 * diagonal_values)
    return corrected_value


def detect02(source, weights="crowdhuman_yolov5m.pt", img_size=1920, conf_thres=0.25, iou_thres=0.45, device="0",
           view_img=True, save_txt=False, save_conf=False, classes=None, agnostic_nms=False, augment=False,
           project="samples/after", name="exp", exist_ok=False, heads=True, person=False, save_img=True):
    global slave_grid, temp_grid
    source, weights, view_img, save_txt, imgsz = source, weights, view_img, save_txt, img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    useTello = source.startswith('tello')

    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    grid_manager = GridManager(GridConfig())
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    # 소스별 설정
    if webcam:
        # strSource = "webcam"
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif useTello:
        # strSource = "tello"
        tello = Tello()
        tello.connect()
        tello.streamon()
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams('udp://@0.0.0.0:11111', img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        grid_manager.clear()

        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam or useTello:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 텍스트 파일 경로

            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            h, w = im0.shape[:2]
            grid_size_x = 6  # 가로 그리드 개수
            grid_size_y = 5  # 세로 그리드 개수
            rect_width = w // grid_manager.config.grid_size_x
            rect_height = h // grid_manager.config.grid_size_y

            # 객체가 인식될 때
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # 객체 좌표
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh) # 객체 클래스명
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # 프레임에 인식된 객체 표시
                        if heads or person:
                            if 'head' in label and heads:
                                plot_one_box(xyxy, im0, grid_manager, label=label, color=colors[int(cls)], line_thickness=3)
                            if 'person' in label and person:
                                plot_one_box(xyxy, im0, grid_manager, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            plot_one_box(xyxy, im0, grid_manager, label=label, color=colors[int(cls)], line_thickness=3)

                temp_grid = np.zeros((grid_size_y, grid_size_x), dtype=int)
                slave_grid = np.zeros((grid_size_y, grid_size_x), dtype=int)

                grid_manager.update_grid(im0.shape)

                for i in range(grid_manager.config.grid_size_y):
                    for j in range(grid_manager.config.grid_size_x):
                        x1 = j * rect_width
                        y1 = i * rect_height
                        x2 = x1 + rect_width
                        y2 = y1 + rect_height

                        # Draw grid
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 255, 255), 2)

                        # 가중치 설정
                        w0 = 0.9  # 현재 셀의 가중치
                        w1 = 0.1  # 상하좌우 인접 셀의 가중치
                        w2 = 0.0  # 대각선 셀의 가중치

                        # 주변 그리드를 통한 1차 보정
                        corrected_value = calculate_corrected_value(grid_manager.grid_counts, i, j, w0, w1, w2)
                        temp_grid[i, j] = corrected_value
                        slave_grid = np.flip(temp_grid)
                        text_pos = (x1 + 10, y1 + 60)

                        # Color based on corrected value
                        if corrected_value >= 7:
                            color = (0, 0, 255)  # Red
                        elif corrected_value >= 6:
                            color = (0, 165, 255)  # Orange
                        elif corrected_value >= 5:
                            color = (0, 255, 255)  # Yellow
                        else:
                            color = (255, 255, 255)  # White

                        # Add semi-transparent overlay
                        overlay = im0[y1:y2, x1:x2].copy()
                        cv2.rectangle(overlay, (0, 0), (rect_width, rect_height), color, cv2.FILLED)
                        cv2.addWeighted(overlay, 0.2, im0[y1:y2, x1:x2], 0.8, 0, im0[y1:y2, x1:x2])

                        # Draw corrected value
                        cv2.putText(im0, str(round(corrected_value, 2)), text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
                '''
                # 각 그리드별 인식된 객체
                for center_x, center_y in circle_centers:
                    grid_x = int(center_x // rect_width)
                    grid_y = int(center_y // rect_height)
                    if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
                        grid_counts[grid_y, grid_x] += 1

                for i in range(grid_size_y):
                    for j in range(grid_size_x):
                        x1 = j * rect_width
                        y1 = i * rect_height
                        x2 = x1 + rect_width
                        y2 = y1 + rect_height
                        text_position = (x1 + 10, y1 + 60) # 640x480 : x1 + 10, y1 + 25 / 800x600 : x1 + 10, y1 + 30 / 1920x1080 : x1 + 10, y1 + 40

                        # 가중치 설정 (임의값 사용. 테스트 하면서 해봐야 함)
                        w0 = 0.9  # 현재 셀의 가중치
                        w1 = 0.1  # 상하좌우 인접 셀의 가중치
                        w2 = 0.0  # 대각선 셀의 가중치

                        # 주변 그리드를 통한 1차 보정
                        corrected_value = calculate_corrected_value(grid_counts, i, j, w0, w1, w2)
                        test_grid01[i, j] = corrected_value
                        slave_grid = np.flip(test_grid01)
                        test_grid01 = np.flip(grid_counts)
                        cv2.putText(im0, str(round(corrected_value, 2)), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.7,(0, 0, 0), 3)
                        # 640x480 : x1 + 65, y1 + 25
                        # 800x600 : x1 + 90, y1 + 30
                        # 1920x1080 : x1 + 240, y1 + 40

                        # 일정 기준이 넘으면 해당 그리드에 색을 입힌다.(BGR 코드)
                        if corrected_value >= 7:
                            alertcolor = (0, 0, 255) # 위험 - 빨간색
                        elif corrected_value >= 6:
                            alertcolor = (0, 165, 255) # 경고 - 주황색
                        elif corrected_value >= 5:
                            alertcolor = (0, 255, 255) # 주의 - 노란색
                        elif corrected_value > 0:
                            alertcolor = (255, 255, 255)  # 안전 또는 평소 - 흰색
                        else:
                            alertcolor = (255, 255, 255)

                        overlay = im0[y1:y2, x1:x2].copy()
                        cv2.rectangle(overlay, (0, 0), (rect_width, rect_height), alertcolor, cv2.FILLED)
                        alpha = 0.2  # 색불투명도
                        cv2.addWeighted(overlay, alpha, im0[y1:y2, x1:x2], 1 - alpha, 0, im0[y1:y2, x1:x2])
                '''

                if view_img:
                    for i in range(grid_size_y):
                        for j in range(grid_size_x):
                            x1 = j * rect_width
                            y1 = i * rect_height
                            c1 = (x1, y1)
                            x2 = x1 + rect_width
                            y2 = y1 + rect_height
                            c2 = (x2, y2)
                            cv2.rectangle(im0, c1, c2, (255, 255, 255), 2)
                            if useTello:
                                cv2.putText(im0, str(tello.get_battery()), (int(w - 80), 50), cv2.FONT_HERSHEY_PLAIN, 3,[255, 255, 255], 3)

                    '''
                    cv2.imshow(str(p), im0)  # 이미지(프레임) 띄우기
                    if cv2.waitKey(1) == ord('q'):  # q를 눌러 나가기
                        cv2.destroyAllWindows()
                        break
                    '''

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        fourcc = 'mp4v'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        yield im0  # 프레임을 제너레이터로 반환


if __name__ == '__main__':
    detect02("samples/test.jpg")
