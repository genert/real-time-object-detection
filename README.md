
# Real time object detection

Install Anaconda - https://www.anaconda.com/products/distribution

```bash
cd model && curl -L https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights > yolov4.weights && cd ..
conda env create -f environment.yaml --force
conda activate real_time_object_detection
VIDEO_URL="https://cdn-005.whatsupcams.com/hls/it_castelsardo02.m3u8" python -m flask run --host=0.0.0.0
open http://localhost:5000/feed
```

