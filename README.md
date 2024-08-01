## Foundational Image Models: A Leap Forward for Surveillance Video Summarization?
Initial Report of Video Object Detection Project at LAS-SCADS -- Laura Dozal,   Ben Strickson 

**Abstract:** The goal of this project is to lower analyst hours spent on triaging surveillance video content. The current analyst approach consists of an analyst looking for any type of change or activity in a video, typically, a change detection algorithm is implemented for those who can run the model. Either way, we are currently using suboptimal techniques for video triage. This project investigates if we can improve object detection approaches in dashcam video triage. Typically dashcam footage is realistic and challenging; it is noisy, long in length, often has no audio, and has a constantly shifting point of view. We use zero-shot models (general transformers and LMMs) to explore two specific problems found in object detection for dashcam videos by LAS staff. The first is obfuscation or obstruction where cars are given a new id and counted more than once if covered by a passing car. The second problem is identifying an unknown or unconventional vehicle like tuk-tuks or various sized trucks.

### Introduction
The goal of this project is to lower analyst hours spent on triaging video content. Our use case is surveillance video triage, which means that our video data will have the following characteristics: long in length, large quantities of redundant information, and no audio. For our research we used dashcam footage which is a useful proxy for noisy surveillance footage. The expected intelligence outputs will vary by use case. We anticipate a number of different analyses will be required e.g. high level activity video searching, or low level object classification and tracking. Two methods were identified to explore these problem areas. These were activity recognition and object detection. Here, object detection is implemented with general transformer models and Large Multimodal Models (LMMs). Activity recognition, which identifies events being performed in a video input, is explored and saved for future research. 

Object detection is used in computer vision to locate and detect objects of interest in an image or video. It does this by identifying the position and boundaries of objects in an image, and classifies those objects into different categories. This type of analysis plays a crucial role in vision recognition, image classification, and retrieval. Our research explores using a zero-shot transformer model, and hopes to use the two step process in future implementations. Zero-shot models classify objects in an image without the help of new annotated data. These models are pre-trained on image classes and process your new data to generalize what it already knows and identify unseen categories. Zero-shot detection models are used in this research by implementing traditional transformer models such as OWL-VIT and LMMs such as GPT4o.

Research at LAS has shown that current object tracking approaches for noisy surveillance footage perform poorly, suggesting that there is significant room for further research in this area (Figure 1). We know that the recently developed LMM technologies are significantly more powerful at understanding image content than previous algorithms such as YOLOV3, but they were not designed for object tracking tasks. Our research demonstrates a path forward for the integration of LMMs into an object tracking workflow. After reviewing dashcam footage shared with us by LAS researchers, we were able to identify clear edge cases that caused problems for traditional object detection models. These cases are set out below, and it is on these that we will judge our proposed approach:
1.	Object Obfuscation: Cars as objects get a unique id. If they are covered or temporarily hidden by a passing car they are given another unique id (or counted twice).
2.	Unknown / Unconventional Vehicle: Tuk-Tuks, motorcycles, and various sized trucks are mis-labeled if labeled at all.
   
![botsort_objectTracking](https://github.com/user-attachments/assets/eb6a1ba8-72d2-4df6-8429-0ca0913736e0)\
***Figure 1:** Current LAS object tracking approaches for noisy surveillance footage. This approach misses unconventional vehicles such as the tuk-tuk on the left of the image.*

#### Initial Exploration (Airport Video Feed)
We spoke with researchers at LAS who are working with object tracking algorithms for dashcam data, and their approach was chosen due to its suitability for complex surveillance videos. We were introduced to airport video surveillance which we began to annotate by first converting ~800 videos of airport runway and taxi footage into segmented clips using ffmpeg to make them easier to process. Then, we installed and downloaded label-studio to begin annotation to review the videos and understand their qualities. The goal here was to get a diverse set of videos annotated i.e. night, day, crowding, etc. During this time, other researchers at SCADs showed that they were able to summarize structured airport video footage quite well using LMMs and Large Language Models (LLMs). But our intentions are to show a more structured output that might include timestamps, pixel location and a little more detail. For example, if object detection or activity recognition algorithms are applied to the CCTV airport feed we would expect them to isolate objects within all frames from the video and classify them according to a list of behaviors such as car parked or car turning. Below we have given a JSON example of the type of structured output you would hope to receive, you would then move up a level of abstraction, possibly through text summarization to make the text summaries easier to triage by analysts:

![airport_image](https://github.com/user-attachments/assets/c0170eb0-2288-4c6e-a4bd-33366512f0ca)
```Output = 
{video_title; “airport_feed_001”,event_frame_list;[
(0:01, [(plane_stationary_(xx,xy,yy,yx),car_parking_(xx,xy,yy,yx),...]),		
(0:02,[(plane_stationary_(xx,xy,yy,yx),car_parking_(xx,xy,yy,yx),...]),
(0:03,[(plane_stationary_(xx,xy,yy,yx),car_parking_(xx,xy,yy,yx),...]),		
(0:04,[(plane_stationary_(xx,xy,yy,yx),car_parking_(xx,xy,yy,yx),...]),
…]}
```

Our colleagues from the other video analysis group had a process that first pulled features from the image frames pulled from a video. Here they used a cosine similarity threshold to filter out noisy or unnecessary images. These frames are passed to an initial LMM and prompted to identify plane takeoff and landing. These summaries are then sent through an LLM model to get an overall understanding of what’s going on in the video. Although their output was more conversational than structured, prompting the LLM to provide that information proved to be fruitful for our colleagues. You would ultimately anticipate receiving an input similar to the below:

![airport_image_sincajas](https://github.com/user-attachments/assets/f2b0a983-1d0d-445f-b9ef-dc51b04e15d1)
```Output = {
video_title; “airport_feed_001”,threshold_frame_list;[
((0:20,0:30), “This is an airport and we see two planes take off and some cars moving…”),
	((1:18,1:30),”This is an airport and we see one plane take off and some cars moving…”),
	((2:30,2:45),”This is an airport and we see several planes parked and some cars…”)…
]
```
We believe that the above approach is useful for a constrained set of video types; there should be a small number of objects of interest, and there should be no need to disambiguate unique objects. Importantly, this technique also requires you to come up with a threshold number that generalizes well, this is feasible for a static camera feed but is unlikely to work for a feed which has multiple points of view. Given our concerns about the previous approach and our use case, noisy surveillance footage without audio, and our requirement to identify unique objects we looked for an alternative approach. 

### Data Collection and Methods
We started out with three distinct model approaches to solving the problem of surveillance video summarization:
1.	Machine learning activity recognition models; we planned to take a state-of-the-art activity recognition architecture, train it on large-scale dashcam activity recognition datasets and then test it on our novel dashcam dataset 
2.	Large pre-trained object detection models; we planned to take the latest generation of image foundational models (e.g. OWL-VIT, DINOv2, YOLO-NAS), and ask them to identify all of the objects within a given frame.
3.	Large pre-trained prompt-LMM models; we planned to take the most capable of the LMMs, GPT4o which was released in 2024, and ask it to identify all of the objects within a given frame and/or provide summaries of the activities occurring within a frame. 

**Data Collection**\
We were provided with a few videos from LAS and chose one that seemed to be noisy enough to implement our analyses. We chose a one minute video and pulled 1800 images from the video that represented video frames. Each second was represented by 30 image frames and each showed a decent amount of noise. These 1800 images were used in part and in full for our analyses.

**Methods**\
Activity Recognition\
On the first approach, we unfortunately failed to successfully implement several activity recognition codebases. In theory this could have been a successful approach to generalized activity recognition, we know that the principle of transfer learning works in many other domains of machine learning. Given the size and diversity of the datasets we found we believed that for the use case of car surveillance we might have a chance at generalizing a model trained on another dataset (https://github.com/salmank255/Road-waymo-dataset) to our novel data. After several updates and re-works of the code we reached the road block of memory size, given the availability of very large GPUs at SCADS, this suggests to us that the authors had access to some sort of HPC cluster for their model training, we were unable to progress any further with this line of research.

Image Foundational Models\
On the second approach, image foundational models, we devised a two step pipeline which would allow us to deliver an object tracking system. Here we had significantly more success in our implementation. OWL-ViT proved incredibly useful for this task and relatively easy to deploy. It is a compromise between a prompt style LMM e.g. chatGPT and a pre-trained object detection model e.g. YOLOV7. When the model is provided with a prompt, “a photo of a car.”,”a photo of a pedestrian”, it will return bounding box coordinates for those specified labels. Once we had those bounding box coordinates, we were able to segment the frame into sub-frames, we then sent those sub-frames to a second model for further description. For the purposes of SCADS and given the time constraint we used GPT4o  and prompted the model to describe the vehicle in the sub-frame in five words or less. But this task is easily accomplished with an open source alternative, after a comparative evaluation to find the best model. Once those descriptions had been generated, our proposal was to combine that embedded semantic information with a centroid tracking algorithm. We previously identified that standard tracking algorithms found in packages such as OpenCV fail repeatedly due to the problem of occlusion, our hope is that additional semantic information will improve the object resolution phase of the tracking pipeline.

![objectDetection-Tracking](https://github.com/user-attachments/assets/6c16b740-74ff-41c0-b799-710aa8827e76)\
***Figure 2:** Multi-step process with traditional zero-shot model and object tracking* 

Large Multimodal Models\
*Initial Prompt Engineering:* For our third approach, LMMs for object tracking we set up several iterations of pipelines in an attempt to achieve a tool which could usefully summarize our dashcam footage. Initially, we began by implementing a cosine similarity threshold which is a common computer vision step that is used to cut out noise (redundancies, outliers, etc) within the image/video data. Here, sequential image features are compared to each other using cosine similarity to produce a score. Images that have a high enough score to show that there is a significant difference are pulled from the threshold. The cosine similarity approach might not be too fruitful here because we want to make sure we're getting all the new cars, and new positioning of the cars which means that a high threshold has to be employed. We initially set the threshold at .94 but it produced only 60 images out of the 1800 image frames from the video.  While using GPT4o with a cosine similarity threshold on the video clips, the model was able to identify a timestamp and multiple details of cars in the video clips. Here, GPT4o was only able to process 30 images at once. We had to move the cosine similarity threshold higher so that we could get 30 image frames. GPT4o was able to identify 8 cars with timestamps, speed and details, along with a short summary from the prompts '*You are generating a video summary using images from frames. Please identify cars in this video and provide a timestamp and their pixel location within the image frame. If possible, provide details about the car.*', and '*Please provide a summary of this video*'. 

In an attempt to review other LMMs, phi3 was implemented on a sample image from the video frames. 
Here we found that phi3 cannot identify anything when using the prompt, Identify all the cars in the image and their pixel location, including visible vehicles in the background.
It will find cars with the prompt: Identify all the cars in the image. And provide the output:
- There are multiple cars in the image, including a yellow taxi in the foreground and various other vehicles in the background.
It will count cars pretty well with the prompt: Count all the cars in the image.:
- There are at least 10 cars visible in the image.
It will count the cars and describe them separately. Not good at pixel location using *Count all the cars in the image and describe them. If possible include their pixel locations.* :
- There are at least 10 cars visible in the image. They are of various colors including yellow, white, and black. The pixel locations for each car cannot be accurately determined due to the image's resolution and the angle of the camera.

Dashcam data, although noisy, has differences and changes in every frame because the video is constantly moving. So testing all we can is important to try to get the LMM to detect the smallest details possible. Through cosine similarity implementation we found that GPT4o can only handle 30 image frames at a time, so our next step was to batch the 1800 images into sets of 30 to run through the model. In all we had 60 batches with 30 images in each batch. Here, each batch also referred to one second in the video.

#### Object Obfuscation and Unconventional Vehicles Problem Implementation 
We chose a video that showed multiple instances of our problem areas; object obfuscation and unknown/unconventional vehicles. This video was one minute long and we could identify at least three instances of each problem by the 6th second. Because of time and resources, we ran the GPT4o model on the first 7 seconds of the video to see if it could identify the problems annotated by the human eye. 

The summaries for obfuscation show some understanding that cars have been hidden or covered by other objects in the video, but upon reading the outputs it doesn't seem to catch the relevant details to include structured data. The first prompt to process the first 3 seconds is below and each output chunk shows the response for each of those seconds. 
The prompt was:
*Each image represents a new image frame in a video. These image frames represent one second's time in the video. From these image frames: identify the number of unique cars shown and describe them. Also provide the second in the video to mark the time. Then see if any of the unique cars shown are temporarily hidden by other cars or objects within the video.*

*Second round of LLM* \
The initial image summaries were converted to string objects and put through the model to run as an LLM summarizer. Here we asked GPT4o to *"Please summarize in detail the description of these summaries about dashcam footage and vehicle obfuscation."* 

A second pass of running the text summaries through an LLM for the first 3 seconds of the video show that gpt-4o could not fully catch the details found in the initial image summarization.
The summaries describe dashcam footage capturing a street scene with various vehicles identified and their visibility analyzed, but miss crucial timing information and other details. These summaries also show instances of unconventional vehicle types like tanker trunks and three wheelers.

The summaries for Unknown/Unconventional vehicles show that the model can identify that there are these types of vehicles present, but the model cannot identify exactly what these vehicles are, only some small descriptions of what and where they’re displayed. The results for identifying unknown or less conventional vehicles like tuk-tuks or multi-sized trucks are shown below. We explored the first 8 seconds of the video to get a wider range of potentially unconventional types.

The prompt used was *"Each image represents a new image frame in a video. These image frames represent one second's time in the video. From these image frames:identify the number of unique cars shown and describe them. Particularly note if there are any cars or vehicle type objects that appear."*

A second pass of running the text summaries through an LLM for the first 8 seconds of the video show that gpt-4o could not fully catch all of the details, but it does identify unconventional vehicles. Identifying unknown or unconventional vehicles proved to be a better task for GPT4o, although the structured and detailed data was not something it was capable of doing well.

### Results
We believe that zero shot object detection using the latest generation of large pre-trained neural networks (OWL-ViT) represents a huge potential step forwards for generalizable object tracking solutions. We repeatedly found that OWL-ViT was able to perform on par with SOTA in terms of detecting objects within busy scenes. We also demonstrated that OWL-ViT was consistently able to outperform YOLOV7 when it came to unusual objects within frames. The performance of  this comparison model on reasonably common objects such as fuel tankers, unusual trucks, and motorcycle taxis is indicative of just how far short current methods fall. It is also worth highlighting that we believe this approach is significantly better than anything the GPT family of models is able to currently offer; models such as GPT4o are often “lazy” when it comes to detecting far away objects in a busy crowded frame and will simply return the nearest 5-15 objects.

We found that Large Multimodal Models have limited memory capacity (30 frames per call), and come with deployment challenges, but can be useful when implemented with various prompting and multi-modal data (text, image, audio). In our case, the GPT4o model works better than phi3, but still cannot identify timestamps, bounding boxes or all of the cars in one shot.

### Discussion and Future Research
We believe that the feasibility of an LMM object detection for object tracking pipeline has been demonstrated during the work done at SCADS 2024. There are several necessary engineering steps that are required before we can move onto quantitative evaluation. First we will deploy and compare several open source LMMs for the object description component of the pipeline. This is important for deployment on the high side, and should consider several variables such as inference speed, memory requirements, and licensing terms. The next step is a review and selection of a centroid tracking algorithm, we will then attempt to devise a novel version of the chosen algorithm to incorporate semantic embeddings. Finally once these technical challenges have been overcome we will be able to move to the evaluation stage of the project. Specifically the focus here will be on those edge cases where previous approaches have failed e.g. object occlusion and unknown object types. Ultimately we want this to be a collaborative effort with other researchers currently working on object tracking at LAS, our hope is that this work is extended to the stage where evaluation can take place and direct comparisons can be drawn with the work of those researchers.

#### Future Research
Multi-Model Implementation
-	Going forward, we hope to implement the OWL-VIT object detection model with traditional object detection applications that use cosine similarity to dedicate these model’s abilities fully and thoroughly investigate our problem areas.
Cosine Similarity
-	Considerations for using cosine similarity with noisy focused data: "what is the best cosine similarity for dashcam video". Is this the best implementation of suppressing the redundant images? Should we even suppress redundant images?
-	Another question would be to see how many video clips are enough to provide a good summary using the LMM. For example, there is not enough space and processing power to analyze ~120 video clips at once (at a .97 threshold). But it did process 30 images at a .94 threshold. One potential deep dive would be to identify a sweet spot in the threshold/image analysis.

