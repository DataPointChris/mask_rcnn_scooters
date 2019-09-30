# Retraining Mask-RCNN to Recognize Electric Scooters

> ### Project by Chris Birch



## Executive Summary



### Table of Contents

- [Acknowledgments](#acknowledgments)
- [Purpose](#purpose)
- [Image Collection](#image-collection)
- [Data Cleaning / EDA](#data-cleaningeda)
- [Preprocessing and Modeling](#preprocessing-and-modeling)
- [Evaluation](#evaluation)
- [Conclusion / Summary / Recommendations](#conclusion-summary-recommendations)
- [TRANSFER LEARNING](#transfer-learning)
- [Learning Points](#learning-points)
- [Moving Forward](#moving-forward)


### Acknowledgments

This is a HUGE list.  I literally got help from every corner of the internet.  This was a great learning project in all aspects.  Absolutely not possible without the help of the following sources:

[matterport Mask-RCNN model pretrained on COCO dataset](https://github.com/matterport/Mask_RCNN)  
[Complete Guide to Creating COCO Datasets](https://www.udemy.com/course/creating-coco-datasets/)  
[Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)  
[RectLabel](www.rectlabel.com)  
[waspinator Train a Mask R-CNN model on your own data](https://patrickwasp.com/train-a-mask-r-cnn-model-on-your-own-dataset/)  
[Training a Mask R-CNN Model Using the Nucleus Data](https://medium.com/@umdfirecoml/training-a-mask-r-cnn-model-using-the-nucleus-data-bcb5fdbc0181)  
[Instance segmentation: automatic nucleus detection.](https://towardsdatascience.com/instance-segmentation-automatic-nucleus-detection-a169b3a99477)  
[Data Augmentation for Deep Learning](https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9)  
[How to run Object Detection and Segmentation on a Video Fast for Free](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)  
[From raw images to real-time predictions with Deep Learning](https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4)  
[Mask R-CNN for object detection and segmentation [Train for a custom dataset]](https://stackoverflow.com/questions/49684468/mask-r-cnn-for-object-detection-and-segmentation-train-for-a-custom-dataset?noredirect=1&lq=1)  

**I have tried my best to include all of the resources I used.  A learning point I found was definitely to create a bookmarks folder for each project and keep all of my sources stored in there for easy retrieval.**


### Purpose

Since I'm wildly interested in autonomous cars, I figured it would be appropriate to do a project on one of the technologies they use, semantic segmentation of image data to classify objects.
Since semantic segmentation requires every pixel to be classified, the computational requirements grow rapidly when images become large or many, or the number of convolution layers grows in size.  This is a factor to consider when deciding which architecture to use, along with size of images and how much processing power will be available at training and inference.
I chose to only include one class of `scooter` to keep it simple enough to explicitly see the results from the various changes I made throughout the process.

### Image Collection

Data was collected by means of my cell phone.  Images and videos both.  Grabbing the scaled resolution of 768 X 1024 allows for reasonable training time with a mid level 1080ti GPU.

Images were collected into a training, validation, and testing set.  There are approximately 196 annotated training images, 8 annotated validation images, and 29 testing images that are not annotated.

The most time consuming part of the data collection process was hand annotating the images.  I searched through a bunch of programs, but ended up on [RectLabel](www.rectlabel.com), a program made for MacOS and unfortunately only available for MacOS.  I tried VGG like everyone else uses but I found it more cumbersome to use.  The main winner here was my Apple Pencil.  I was able to use Duet to make my ipad another display for my MacBook and then use the my Apple Pencil to draw the polygons used for the annotations.  This saved me tons of time and added accuracy compared to clicking with a mouse.

### Data Cleaning / EDA

The beauty of image data is that there isn't much data cleaning or EDA to do aside from making sure that images are being read correctly, are coming in as the correct datatype, and the annotations match.
There are a few cells included in the notebook to display the training images with masks for a sanity check.

### Preprocessing and Modeling

Using a pretrained, already set up model makes the preprocessing step much simpler, but also harder to modify to specific needs.  Since all of my images for this project were coming it at the same resolution, I didn't need to include any zero-padding or resizing.

Originally I was going to implement the Keras `ImageDataGenerator()` to augment my small dataset of images, but when I was snooping around I found that they already had one implemented!  Score!  This method was a more simple setup.  Just set `augmentation=augmentation` in the `.fit` method and speficy the augmentation parameters.  This model uses the `imgaug` library which has many options available, tons actually.  One problem I did come across was having to set `_to_deterministic()` on the first augmentation so the masks received the exact same augmentation as the original images, otherwise the annotations would no longer match.

That's it!  Decide on the number of epochs to run, batch size that your GPU can handle, and let it go.

### Evaluation

As mentioned before, when using a pretrained model already built out, you lose some degree of control.  An understandable tradeoff with the amount of work they put into making this a great base pretrained model for many other applications.
What I found was that even with image augmentation, my model didn't seem to do better with more epochs.  At around 200 it was doing the best it could.  I was curious how it would turn out if I trained it on all layers instead of just the head layers, and found out once again having a *tiny* dataset really contributes to overfitting.  I included a few images of bicycles and mopeds along with the scooters for testing.
One thought was that the model could discern the difference between the scooters and scooters by having them in the same images with the scooters only labeled, the other being that I wanted to test the model's ability to recognize scooters and not mopeds.
{Here's the result}
Unfortuately, some degree of freedom and understanding was lost using their more complicated model and predefined everything, so that I wasn't able to view the history very well, and made the rookie mistake to not set the model to store the history in a variable.

### Conclusion / Summary / Recommendations

This is a great model overall to get results quickly.  Even though I learned more than I could imagine using this model, for better understanding of how the model reacts to more or less images, augmentation, more steps per epoch or more epochs, I would try training a more basic model with just the images, or a more straightforward example that can use pretrained weights, but has a more accessible way to configure all of the parameters.  
I think this model could be used for many differnt purposes, if trained correctly.  I already have some other ideas that I might try if I can get my scooter model working better.

## TRANSFER LEARNING

Transfer learning applies both to the model and myself.
Using a pretrained model is the only option in some cases, whereas others it's just the smart option.  If I have some nails to hammer in, I can go and forge a hammer myself, or I could use a premade hammer that is great at it's job and has research and theory behind it, so that I'm able to use my time and energy to hit the nails firm and square.
I could probably build a comparable model to this one, with MONTHS of time spent, but I only had three weeks for this project and this was a very efficient and practical method.

On the other side of that, I had thousands of datapoints of transfer learning going directly into my cranium and absorbing into my brain.  A great side effect of taking a complicated model with well documented code is the ability to learn by example and have a framework to use in order not to get lost too often along the way.

### Learning Points

 - Reading over an advanced programmer's code helps immensly with visualizing how code should be structured and broken into functions and classes
 - Playing with JSON files and figuring the structure will be a very useful skill
 - Image augmentation is a great way to increase the size of a dataset artificially without negative consequences
 - The tradeoff of cost/time/ease of creating and using a model are well understood when a model has over 100 layers and takes HOURS to train.  Everything needs to be thought out and planned better before wasting time and money on useless training.
 - The BEST way to learn anything is to experiment and find all ways to break it then figure out how to fix it.

### Moving Forward

[ ] Using a trained model to help annotate more images. 

[ ] Try retraining a base model with thousands of annotated images.

[ ] Experiment with color and lighting image augmentation that can't be applied to masks.

[ ] View model training history from start to finish.

[ ] Possibly include another class. 

[ ] Try training model from scratch with COCO dataset to get a feel for handling very large image dataset

[ ] Implement this model in a real time video.

[ ] Implement a variation in an autonomous {enter world destroying robot here}




