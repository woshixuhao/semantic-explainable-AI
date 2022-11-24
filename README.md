# semantic-explainable-AI
The codes and dataset for the semantic explainable AI (S-XAI).
The dataset utilized in this work can be downloaded from:
https://drive.google.com/file/d/1OnEX67h-C7Q0_Tul_3TYousoTRW_Xd69/view?usp=sharing
The VGG_224.pth can be obtained by the CNN_training or downloaded from:
https://drive.google.com/file/d/1VshtrPKOm4e5czbjAjQKJjtWZPRBRJN0/view?usp=share_link


# Some notes for using the cut_position function
The cut_position function is utilized to mask the semantic position manually. For each picture, we should click 4 times on the semantic position. You will see four red dots on the picture. Then, the picture will be closed and reopened automatically. You need to click 1 time on a nearby position (e.g., body) to decide the color of mask. When the operation is finished, the next picture will be opened. If the picture donnot has semantic position, you can click 4 time on the same place to skip this picture. 


# Files that can generate automatically
Before running the main.py, we should run the GA.py to conduct lime and select the best superpixels combination. The results are generated in lime_save dir.
