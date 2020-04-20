# face_recognition
recognize face with marchine learning

# 1. Install pipenv ( or virtualenv)
$ pip install pipenv 

# 2. Run python3 in pipenv
$ pipenv --python python3

# 3. Install package requests, face_recognition, numpy, opencv
$ pipenv install requests
$ pipenv install face_recognition
$ pipenv install numpy
$ pipenv install opencv-python

Note : If opencv has error not found version, do following steps : 
$ brew install opencv
$ ln -s "$(brew --prefix)"/lib/python3.6/site-packages/cv2*.so "$(pipenv --venv)"/lib/python3.6/site-packages
$ pipenv run python -c "import cv2; print(cv2.__version__)"

# 4. Spawns a shell within the virtualenv.
$ pipenv shell

# 5. To run program 
$ python <file-name> 