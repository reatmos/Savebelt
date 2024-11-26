# Savebelt
Design of a Crowd Density Analysis Service Using Grid-Based on Master/Slave Architecture



https://github.com/user-attachments/assets/52f89bfc-bbd0-40d9-8088-d76b84d9be45


## Requirements
- [crowdhuman_yolo5m.pt](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view)

- Files in samples/

## Library Versions
- Python 3.8.10

- torch==2.3.0+cu121,
- torchvision==0.18.0+cu121,
- torchaudio==2.3.0+cu121
- numpy version to 1.24.4

[For cmd]
- pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

[For PyCharm]
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## How to Run
- Run main.py and check in service.html

[Optional]
- To run each camera separately, uncomment the following code in `if view_img:` in detect(n), comment out `yield im0`, specify source in `if __name__ == '__main__':`, and execute

```
  cv2.imshow(str(p), im0) # show image (frame)
  if cv2.waitKey(1) == ord('q'): # press q to exit
    cv2.destroyAllWindows()
  break
```

## Notes
- [decthead](https://github.com/mehdighasemzadeh/Crowd-Counting-YOLOV5)
