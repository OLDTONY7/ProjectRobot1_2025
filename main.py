import sensor, image, time, lcd, gc
from maix import KPU, GPIO, utils
from fpioa_manager import fm
from board import board_info
from modules import ybserial
import uhashlib


serial = ybserial()


lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=100)
clock = time.clock()


feature_img = image.Image(size=(64, 64), copy_to_fb=False)
feature_img.pix_to_ai()


FACE_PIC_SIZE = 64
dst_point = [
    (int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
    (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
    (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
    (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
    (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112))
]


anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875,
          0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875,
          0.4721875, 0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)

kpu = KPU()
kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
kpu.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240, net_w=320, net_h=240,
               layer_w=10, layer_h=8, threshold=0.7, nms_value=0.2, classes=1)


ld5_kpu = KPU()
ld5_kpu.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")


fea_kpu = KPU()
fea_kpu.load_kmodel("/sd/KPU/face_recognization/feature_extraction.kmodel")


start_processing = False
BOUNCE_PROTECTION = 50
fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
def set_key_state(*_):
    global start_processing
    start_processing = True
    time.sleep_ms(BOUNCE_PROTECTION)
key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)


def hash_feature(feature, salt="face_salt_01"):
    feature_str = ",".join(["%.4f" % v for v in feature])
    salted = salt + feature_str
    sha = uhashlib.sha256()
    sha.update(salted.encode())
    return sha.digest()


record_ftrs = []
record_hashes = []
THRESHOLD = 80.5
recog_flag = False
msg_ = ""


def extend_box(x, y, w, h, scale):
    x1 = max(1, int(x - scale * w))
    x2 = min(319, int(x + w + scale * w))
    y1 = max(1, int(y - scale * h))
    y2 = min(239, int(y + h + scale * h))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1

# 主循环
while True:
    gc.collect()
    clock.tick()
    img = sensor.snapshot()
    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()
    fps = clock.fps()

    if len(dect) > 0:
        for l in dect:
            x1, y1, cut_img_w, cut_img_h = extend_box(l[0], l[1], l[2], l[3], scale=0)
            face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
            face_cut_128 = face_cut.resize(128, 128)
            face_cut_128.pix_to_ai()

            out = ld5_kpu.run_with_output(face_cut_128, getlist=True)
            face_key_point = []
            for j in range(5):
                x = int(KPU.sigmoid(out[2 * j]) * cut_img_w + x1)
                y = int(KPU.sigmoid(out[2 * j + 1]) * cut_img_h + y1)
                face_key_point.append((x, y))

            T = image.get_affine_transform(face_key_point, dst_point)
            image.warp_affine_ai(img, feature_img, T)
            feature = fea_kpu.run_with_output(feature_img, get_feature=True)


            scores = []
            for j in range(len(record_ftrs)):
                score = kpu.feature_compare(record_ftrs[j], feature)
                scores.append(score)

            if len(scores):
                max_score = max(scores)
                index = scores.index(max_score)
                if max_score > THRESHOLD:
                    img.draw_string(0, 195, "person:%d,score:%2.1f" % (index, max_score), color=(0, 255, 0), scale=2)
                    recog_flag = True
                else:
                    img.draw_string(0, 195, "unregistered", color=(255, 0, 0), scale=2)


            if start_processing:
                record_ftrs.append(feature)
                record_hashes.append(hash_feature(feature))
                print("record_ftrs:%d" % len(record_ftrs))
                start_processing = False


            if recog_flag:
                img.draw_rectangle(l[0], l[1], l[2], l[3], color=(0, 255, 0))
                recog_flag = False
                msg_ = "Y%02d" % index
            else:
                img.draw_rectangle(l[0], l[1], l[2], l[3], color=(255, 255, 255))
                msg_ = "N"

            del face_cut_128
            del face_cut

    if len(dect) > 0:
        send_data = "$08" + msg_ + ',' + "#"
        time.sleep_ms(5)
        serial.send(send_data)
    else:
        serial.send("#")

    img.draw_string(0, 0, "%2.1ffps" % fps, color=(0, 60, 255), scale=2.0)
    img.draw_string(0, 215, "press boot key to regist face", color=(255, 100, 0), scale=2.0)
    lcd.display(img)


kpu.deinit()
ld5_kpu.deinit()
fea_kpu.deinit()