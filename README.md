# Video-Classification-in-Real-Time

Video classification using VGG16 as a feature extractor and seasoning with RNN. Dataset used is UCF101 (Cricket bowling and batting classes).

> A simple RNN is used to better classify temporal frame sequences from videos.

Cricket bowling             |  Cricket batting
:-------------------------:|:-------------------------:
![](https://github.com/saimj7/saimj7/blob/master/dino.gif)  |  ![](https://github.com/saimj7/saimj7/blob/master/dino.gif)

--- 

## Table of Contents
* [Background Theory](#background-theory)
* [Running Inference](#running-inference)
* [Pipeline](#pipeline)
  - [Preprocessing](**preprocessing**)
  - [Training](**training**)
* [References](#references)
* [Next Steps](#next-steps)

## Background Theory
**Feature extraction:**
- Pretrained VGG16 is used as a feature extractor after fine tuning/unfreezing its 4 top layers.
- A simple classifier is then connected to VGG16 and trained to identify if the frame belongs to class1 or 2. Then the top classifier is disconneceted and only dense layer with 1024 output size is used to obtain the sparse representations of each frame. 
- Data to lstm format: For each video frame, the sparse representations are stacked into a tensor of size (NUM_FRAMES, LOOK_BACK, 1024). 

---
**RNN:**
- A standard LSTM is used. Note that you need GPU/CUDA support to run CUDnnLSTM layers in the model. Finally the LSTM network is trained to distinguish between your desired class1 and 2 videos.

## Running Inference
- Install all the required Python dependencies:
```
pip install -r requirements.txt
```
- To run inference either on a test video file or on webcam: 
```
python run.py 
```
> Note that the inference is set on test video file by default. To change it, adjust the config. options at the start of run.py
- Trained model weights (for this example) can be downloaded from here: . Make sure you extract them into the folder 'weights'.

## Pipeline

***Preprocessing:***
- Some image processing is required before training your own data. In Preprocessing.ipynb file, the frames from each video classes are extracted and sorted into respective folders.
- Note that the frames are resized to 224x224 dimensions (which is VGG16 input layer size).

***Training:***
- Train.ipynb 

## References

***Main:***
- coming soon

***Optional:***

## Next steps
- coming soon

<p>&nbsp;</p>

---

## Thanks for the read & have fun!

> To get started/contribute quickly (optional) ...

- **Option 1**
    - 🍴 Fork this repo and pull request!

- **Option 2**
    - 👯 Clone this repo:
    ```
    $ git clone https://github.com/saimj7/People-Counting-in-Real-Time.git
    ```

- **Roll it!**

---

saimj7/ 06-09-2020 © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.