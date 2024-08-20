from ultralytics import YOLO

model = YOLO('runs/FLDet/test_UAVDT_FLDet-S/weights/best.pt')
# model = YOLO('runs/FLDet/test_VisDrone_FLDet-S/weights/best.pt')

model.predict('test_image/visiual_uavdt', save=True, show_labels=False, show_conf=True, line_width=2)
# model.predict('test_image/visiual_visdrone', save=True, show_labels=False, show_conf=False, line_width=2)
