
import sensor,image,lcd,time
import KPU as kpu

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
#sensor.set_hmirror(1)
#sensor.set_vflip(1)
sensor.set_windowing((224, 224))
sensor.set_brightness(2)
#sensor.set_contrast(-1)
#sensor.set_auto_gain(1,2)

sensor.run(1)
clock = time.clock()
classes = ['class_1']
task = kpu.load(0x300000)
anchor = (1, 1.2, 2, 3, 4, 3, 6, 4, 5, 6.5)
a = kpu.init_yolo2(task, 0.17, 0.3, 5, anchor)
while(True):
    clock.tick()
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    print(clock.fps())
    if code:
        for i in code:
            a=img.draw_rectangle(i.rect())
            a = lcd.display(img)
            print(i.classid(),i.value())
            for i in code:
                lcd.draw_string(i.x(), i.y(), classes[i.classid()], lcd.RED, lcd.WHITE)
                lcd.draw_string(i.x(), i.y()+12, '%f1.3'%i.value(), lcd.RED, lcd.WHITE)
    else:
        a = lcd.display(img)
a = kpu.deinit(task)
