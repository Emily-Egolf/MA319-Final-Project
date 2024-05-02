# MA319-Final-Project
## MLjong
MLjong is a reinforcement learning ai to play riichi mahjong using the mjx mahjong
framework and tensorflow

## Setup
 - Install all requirements listed in requirements.txt
 - Download Tenhou log from [here](https://drive.google.com/file/d/1K3-WbKncWhsu1OyatveGemdyMrz90A9Q/view?usp=drive_link)
 - Create initial model
 ```python discard_model.py --dataset_path DATASET_PATH --cnn_path CNN_OUTPUT_PATH```
 - Run reinforcement training
 ```python mljong.py --discard_model_path CNN_OUTPUT_PATH```
