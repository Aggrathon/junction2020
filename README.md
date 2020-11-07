# D`AI`rector

This is a project for [Junction 2020](https://connected.hackjunction.com), targeting the Genelec challenge. A video demonstrating the project can be viewed [here](https://youtu.be/fmVov_BnvsQ).


## Challenge

Create something that enables the capture of a 360 video of yourself (with a mobile phone) without the help of anyone and without moving.

One solution would be to mount you phone somewhere, and then film yourself turning on the spot. But the question then becomes: which spot? It's difficult to frame yourself when you cannot see the screen. Then you also need to cut away all the extra footage.


## Solution

My solution is to introduce AI that can guide you through the process and direct you (through audio) to the sweet spot. It can also edit the video to improve the framing, cut the video, and defocus the background.


## Technology

The main AI model is a pose estimator. Here I use [BlazePose](https://google.github.io/mediapipe/solutions/pose) from Google that is designed to work on phones. This way I can calculate how the subject should move so that the head and shoulders fit in the videoframe (with margins).

I also use an image segmentation model, [Deeplab v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1), for separating the background from the subject during the post-processing. An additional AI model that I considered, but so far have not implemented, is [portrait shadow manipulation](https://people.eecs.berkeley.edu/~cecilia77/project-pages/portrait.html), to remove shadows in case the lighting is insufficient.

Finally I also use text-to-speech for giving directions to the user. Here AI could be used as well for producing a natural sounding voice, but for now I use a very basic one ("good enough" for a demo). An alternative would be to prerecord the voice-lines (by a human).

The components have been chosen such that they would work well on a mobile phone (e.g. the neural networks are designed for that). However, I haven't (yet) integrated them into a mobile app, since I deemed it to time-consuming / risky for a weekend hackathon.


## Usage

The project is implemented using Python and the main dependencies are `mediapipe`, `opencv`, `tensorflow`, `pyttsx3`, and `numpy`. To capture a 360 video you also need a camera (accessed via `opencv.VideoCapture(0)`). With these prerequisites prepared, just run `python head_director.py` to start the AI guidance and 360 video capture.
