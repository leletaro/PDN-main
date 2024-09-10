import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from scipy.spatial import distance

# 加载模型
model = YOLO('PdamageDetection.pt')

# 视频路径
video_path = "dark1.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频画布的计数区域
counting_regions = [
    {
        "name": "Region",
        "polygon": Polygon([(0, frame_height/2), (0, frame_height /3 ), (frame_width, frame_height /3), (frame_width, frame_height/2)]),  # 多边形点覆盖整个画布
        "counts": 0,
        # "is_transparent": True, 
        "region_color": (205, 205, 180),  
        "text_color": (0, 0, 0),  # 文字颜色
    },
]

 
  
# 类别计数字典
class_counts = {}
# 存储跟踪对象
tracked_objects = {}  # 使用字典按区域和类别跟踪对象
object_ids = {}  # 物体ID
next_id = 1  # 下一个物体ID
distance_threshold = 50  # 距离阈值，用于判断是否为同一物体

while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    
    # 使用 YOLOv8 模型进行预测
    results = model.predict(source=frame)
    result = results[0]

    # 获取检测到的类别信息
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # 获取检测框的坐标信息
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 提取坐标值并转换为numpy格式
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            point = Point(x_center, y_center)

            # 检查中心点是否在任何定义的区域内
            for region in counting_regions:
                if region["polygon"].contains(point):
                    cls = int(box.cls.cpu().numpy())  # 获取类别索引

                    # 初始化区域和类别的跟踪列表
                    if region["name"] not in tracked_objects:
                        tracked_objects[region["name"]] = {}

                    if cls not in tracked_objects[region["name"]]:
                        tracked_objects[region["name"]][cls] = []

                    # 进行物体跟踪
                    best_match_id = None
                    min_dist = float('inf')
                    for obj_id, obj in tracked_objects[region["name"]][cls]:
                        # 计算当前物体与已跟踪物体之间的距离
                        dist = distance.euclidean((x_center, y_center), (obj['x_center'], obj['y_center']))
                        if dist < distance_threshold and dist < min_dist:
                            min_dist = dist
                            best_match_id = obj_id
                    
                    if best_match_id is not None:
                        # 更新已有物体的中心点
                        for obj in tracked_objects[region["name"]][cls]:
                            if obj[0] == best_match_id:
                                obj[1]['x_center'] = x_center
                                obj[1]['y_center'] = y_center
                                break
                    else:
                        # 新物体
                        obj_id = next_id
                        next_id += 1
                        tracked_objects[region["name"]][cls].append((obj_id, {'x_center': x_center, 'y_center': y_center}))
                        # 更新类别计数
                        if cls in class_counts:
                            class_counts[cls] += 1
                        else:
                            class_counts[cls] = 1
                        region["counts"] += 1  # 更新区域内的物体计数
                    break

    # 在帧上绘制预测结果和计数区域
    anno_frame = result.plot()
    for region in counting_regions:
        # 绘制区域的多边形
        pts = np.array(region["polygon"].exterior.coords, np.int32)
        cv2.polylines(anno_frame, [pts], isClosed=True, color=region["region_color"], thickness=2)
        # 显示区域内的计数
        cv2.putText(anno_frame, f'{region["name"]} Count: {region["counts"]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, region["text_color"], 2)

    # 显示视频帧
    cv2.imshow("山东大学路面病害分析系统", anno_frame)

    # 设置延迟以减慢视频播放速度
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 打印类别计数
print("Detected object counts by class:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count}")
