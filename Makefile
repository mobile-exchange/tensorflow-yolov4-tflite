.PHONY: convert2tf detect install

convert2tf: 
	python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny --input_size 640 --model yolov4 --tiny

detect:
	python detect.py --weights ./checkpoints/yolov4-tiny --size 640 --model yolov4 --image ./data/kite.jpg --tiny

install:
	python -m pip install -r requirements.txt
