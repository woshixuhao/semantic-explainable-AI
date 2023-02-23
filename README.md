# semantic-explainable-AI
The codes and dataset for the semantic explainable AI (S-XAI).
The dataset utilized in this work can be downloaded from:
https://drive.google.com/file/d/1OnEX67h-C7Q0_Tul_3TYousoTRW_Xd69/view?usp=sharing  
The VGG_224.pth can be obtained by the CNN_training or downloaded from:
https://drive.google.com/file/d/1VshtrPKOm4e5czbjAjQKJjtWZPRBRJN0/view?usp=share_link  


# Note
If the program raise the error that do not find certain file, please run some functions first to generate relevant files.
For example, if there is error in the S-XAI.py, one can try these functions to generate required files.
·get_inverse_position(conv_out_2_0)  
·cut_position(use_model)  
·sort_index_origin,sort_index_position,space_index,space_value=get_position(conv_out_2_0,5,show_picture=True)  
·plot_distribution(space_index,space_value,picture='small')  
·print_word(conv_out_2_0,512)  
·val_distribution(conv_out_2_0,512)  
·cat_space,cat_space_pic=get_space(conv_out_2_0,512,sort_index_origin,sort_index_position,add_position='position')  

# Some notes for using the cut_position function
The cut_position function is utilized to mask the semantic position manually. For each picture, we should click 4 times on the semantic position. You will see four red dots on the picture. Then, the picture will be closed and reopened automatically. You need to click 1 time on a nearby position (e.g., body) to decide the color of mask. When the operation is finished, the next picture will be opened. If the picture donnot has semantic position, you can click 4 time on the same place to skip this picture. 


# Files that can generate automatically
Before running the main.py, we should run the GA.py to conduct lime and select the best superpixels combination. The results are generated in lime_save dir.
