# Machine Learning Project

The following is a machine learning project using methods learned in the Machine Learning course led by Assad Sayeed at the University of Gothenburg

## Introduction
The project is an experiment with sentiment analysis. The goal is to accurately predict the genre of a song based on the contents of the lyrics using pretrained and learned vectors. The model is trained on a dataset containing song lyrics and their genre annotation. I managed to gain accuracy of around 46/47 percent, pretrained vectors didn't make much difference. I believe preprocessing may be to blame as the dataset contains examples from multiply languages. 

## How to run

The program can be used by running the script `main.py`. As I have not found a way to export Torchtext generators unfortunately we have to pre_process, train and test in the same script. It doesn't take that long however. The command line arguments ar as follows:

1. `-M` : This argument allows you to name the model as the results will be exported to `graphs` folder (string)
2. `-B` : This allows you to define the size of the batches (int)
3. `-E` : This is the number of Epochs for training (int)
4. `-P` : This allows you to specify whether or not to use pretrained vectors or not. (y/n) 
5. `-D` : If using pretrained this allows you to select dimension size (50,100,200,300)

## Write up

You can find the write up in the pdf file. It is not yet complete but I thought I would put it there so you can see how I am getting on and maybe you have feedback. 
