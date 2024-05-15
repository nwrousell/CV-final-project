# Computer Vision Final Project

- EAST [paper](https://arxiv.org/pdf/1704.03155.pdf) [re-implementation githhub reference](https://github.com/SakuraRiven/EAST)
- [Preprocessing images](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)
- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)
- [Overview of 2022 SOTA in text recognition](https://dilithjay.com/blog/sota-in-scene-text-recognition-2022)

## Datasets
### Text Detection
- [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)

### Text Recognition
- [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)
- [SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

### General Approach
1. Image input
2. Convert to grayscale/high contrast
3. Character detection
4. Character classification
5. Use google translate api
6. Put the text back



## Results

Test detection with EAST and ORB

https://github.com/nwrousell/CV-final-project/assets/36180122/311b2601-12be-4e91-8e1e-32c2cd2338d9


Text recognition with VitSTR and ORB

https://github.com/nwrousell/CV-final-project/assets/36180122/ce207c57-f6e0-460f-a1e3-27bec9cff4ef


