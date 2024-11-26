import threading

def socket01():
    from camera01 import app01
    app01.run(host='0.0.0.0', port=5000) # 웹에서 받을 포트

def socket02():
    from camera02 import app02
    app02.run(host='0.0.0.0', port=5001) # 웹에서 받을 포트

def socket03():
    from camera03 import app03
    app03.run(host='0.0.0.0', port=5002) # 웹에서 받을 포트

def socket04():
    from textframe import app04
    app04.run(host='0.0.0.0', port=5003) # 웹에서 받을 포트

if __name__ == '__main__':
    # 멀티 쓰레드 작업
    t1 = threading.Thread(target=socket01)
    t2 = threading.Thread(target=socket02)
    t3 = threading.Thread(target=socket03)
    t4 = threading.Thread(target=socket04)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()