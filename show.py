import aedat
import cv2
import numpy as np

decoder = aedat.Decoder("/home/tlab4/zjg/data/2023127/event7.aedat4")

index_frame = 0
# index_events = 0
# height = 480
# width = 640
for packet in decoder:

# # 现在是每个灰度图之间的事件太少，导致难以识别
#     if "events" in packet:
#         event_img = np.zeros((height, width))
#         events = packet["events"]
#         for event in events:
#             x = event[1]
#             y = event[2]
#             if event[3]:
#                 p = 1
#             else:
#                 p = -1
#             event_img[y, x] += p
            
#         event_img_min, event_img_max = event_img.min(), event_img.max()
#         event_img_scaled = (event_img - event_img_min) / (event_img_max - event_img_min) * 255
#         event_img = event_img.astype(np.uint8)
#         if index_frame > 3000:
#             cv2.imshow(f"photo", event_img)
#             cv2.waitKey(1)
#         index_frame += 1
    if "frame" in packet:
        index_frame += 1
        image = packet["frame"]["pixels"]
        image_ts = packet["frame"]["t"]
        # image_begin_ts = packet["frame"]["begin_ts"]
        # image_end_ts = packet["frame"]["end_ts"]
        # image_list.append(image)
        # image_list_ts.append(image_ts)
        # index_frame += 1
        if index_frame == 2000:
            cv2.imwrite(f"{index_frame}.png", image)
        # cv2.imshow("Photos", image)
        # cv2.waitKey(1)


