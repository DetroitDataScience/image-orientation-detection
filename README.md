# Image Orientation Detection
A presentation and exercise designed as an introduction into machine learning.  Presented at the [Detroit Data Science Meetup on 3/30/2018](https://www.meetup.com/Detroit-Data-Science-Meetup/events/238413214/).

# Getting Started

1. Clone the repository:

    ```
    git clone git@github.com:DetroitDataScience/image-orientation-detection.git
    ```

2. The training and test data is pretty large and you probably don't need it right now.  However, it is stored in the image-orientation-detection-data git repository.

    ```
    git clone git@github.com:DetroitDataScience/image-orientation-detection-data.git ./image-orientation-detection/data
    ```

3. Install [python 2.7.x](https://www.python.org/downloads/) if you do not already have it.  You can run `python --version` to see if python is already installed.

4. Install the python dependancies found in requirements.txt

    ```
    pip install -r requirements.txt
    ```

5. Run the `student.py` application

    ```
    python student.py
    ```