.PHONY: convert2tf detect install

convert2tf: 
	python3 save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny --input_size 640 --model yolov4 --tiny

detect:
	python3 detect.py --weights ./checkpoints/yolov4-tiny --size 640 --model yolov4 --image ./data/kite.jpg --tiny

train:
	python3 train.py --weights ./data/yolov4-tiny.weights --model yolov4 --tiny

install:
	python3 -m pip install -r requirements.txt
