# Mount folder path sesuai yang diinginkan
# Hati-hati, banyak `path` yang hardcoded di codingan `datacleaning_arcface.py` & `core_arcface_cleaning.py`

nvidia-docker run --rm -it -d\
-v /home/luqmanr/.Xauthority:/home/luqmanr/.Xauthority \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-e DISPLAY=${DISPLAY} \
--device=/dev/video0:/dev/video0 \
-p 5000:5000 \
--shm-size=12g \
--name=dataset_cleaning \
--privileged \
-v /mnt/Data/RKB-Face-Git/FaceCleaner:/workspace/ \
-v /mnt/Data/:/mnt/Data/ risetai/research:face-recognition
