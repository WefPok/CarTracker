import cv2
import os, time
import numpy as np
from sort.sort import Sort
from openvino.inference_engine import IECore

# Initialize SOR
max_age = 20  # increase max_age to 5
min_hits = 5  # increase min_hits to 5
iou_threshold = 0.5  # increase iou_threshold to 0.5

# Initialize SORT with custom parameters
mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

# [your existing code to load the model, open the video file, etc.]
# load the model
ie = IECore()

# Read the model
model_xml = "intel/vehicle-detection-0202/FP32/vehicle-detection-0202.xml"  # replace with your actual path
model_bin = os.path.splitext(model_xml)[0] + ".bin"
net = ie.read_network(model=model_xml, weights=model_bin)

# Prepare blobs
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape
exec_net = ie.load_network(network=net, device_name="CPU")

# Open video file or capture device
cap = cv2.VideoCapture('vid.mp4')

# Get the video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('out.mp4', fourcc, 20.0, (frame_width, frame_height))

start_time = time.time()
frame_count = 0

while (cap.isOpened()):

    ret, image = cap.read()
    if ret == True:
        frame_count += 1
        # [your existing code to perform inference and get bounding boxes]
        resized_image = cv2.resize(image, (w, h))
        input_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, 0)

        # perform inference
        res = exec_net.infer(inputs={input_blob: input_image})

        # Prepare the bounding boxes for SORT
        # Each row in the array represents a bounding box in the format (x1, y1, x2, y2)
        bounding_boxes = []
        for obj in res[out_blob][0][0]:
            if obj[2] > 0.2:
                xmin = int(obj[3] * frame_width)
                ymin = int(obj[4] * frame_height)
                xmax = int(obj[5] * frame_width)
                ymax = int(obj[6] * frame_height)
                bounding_boxes.append([xmin, ymin, xmax, ymax])

        bounding_boxes = np.array(bounding_boxes)
        # Update SORT with the new bounding boxes
        if len(bounding_boxes) > 0:
            tracked_objects = mot_tracker.update(bounding_boxes)
        else:
            tracked_objects = []
        # Draw the tracked objects
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, obj_id = map(int, obj)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, str(obj_id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # [your existing code to write the frame to the output video file]

        out.write(image)
    else:
        break

end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time

print('Processed {} frames in {:.2f} seconds for an FPS of {:.2f}.'.format(frame_count, elapsed_time, fps))

cap.release()
out.release()
cv2.destroyAllWindows()

# [your existing code to release everything and close windows]
