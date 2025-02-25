from psychopy import visual, core, event
import numpy as np
import random

##Preparing the data:
'''
collecting 500 words to be tested on and the ration of correct vs incorect will be a parameter
'''
correct_word_num = 250
incorrect_advanced_num = 150
incorrect_trick_num = 100

##Randomly get subset of these words from the txt file. 
def select_random_words(file_path, num_words=250):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file if line.strip()]
    
    if len(words) < num_words:
        raise ValueError("The file contains fewer words than requested.")
    
    random_words = random.sample(words, num_words)
    
    return random_words

correct_word_path = "correct.txt" 
correct_words = select_random_words(correct_word_path, num_words = correct_word_num)
correct_words = [(word, 1) for word in correct_words]

incorrect_advanced_path = "incorrect_advance.txt"  
incorrect_advanced_words = select_random_words(incorrect_advanced_path, num_words = incorrect_advanced_num)
incorrect_advanced_words = [(word, 0) for word in incorrect_advanced_words]

incorrect_trick_path = "incorrect_trick.txt"  
incorrect_trick_words = select_random_words(incorrect_trick_path, num_words = incorrect_trick_num)
incorrect_trick_words = [(word, 0) for word in incorrect_trick_words]

# Combine all lists into one
combined_list = correct_words + incorrect_advanced_words + incorrect_trick_words

# Shuffle the combined list
random.shuffle(combined_list)




# Set up the window
win = visual.Window(size=[800, 600], color="black", units="norm")

# Define the word and flicker/display parameters
# word = "table"
flicker_freq = 12          # Flicker frequency in Hz
base_duration = 0.5        # Duration per character in seconds
extra_duration = 0.2       # Extra duration for the last character
frame_rate = 60            # Assumed monitor refresh rate

# Create a constant square frame stimulus
square_frame = visual.Rect(
    win,
    width=0.4,  # Adjust size as needed
    height=0.4,
    pos=(0, 0),
    lineColor="white",
    fillColor=None,
    lineWidth=2
)

# Prepare a text stimulus for displaying letters
text_stim = visual.TextStim(
    win,
    text="",
    pos=(0, 0),
    color="white",
    height=0.2
)

# Prepare a prompt screen stimulus (to be shown after all letters)
prompt_text = visual.TextStim(
    win,
    text="Press RIGHT arrow if you identified the vocabulary\nPress LEFT arrow if not",
    pos=(0, 0),
    color="white",
    height=0.07
)

label_list = []
user_response_list = []
for word, label in combined_list:
    label_list.append(label)
    # Display each character in the word sequentially
    for i, letter in enumerate(word):
        # Determine the display duration: last letter gets extra time
        duration = base_duration + extra_duration if i == len(word) - 1 else base_duration
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime()
            # Flicker logic: show the letter when the sine wave is positive, otherwise blank
            if np.sin(2 * np.pi * flicker_freq * t) > 0:
                text_stim.text = letter
            else:
                text_stim.text = ""
            
            square_frame.draw()
            text_stim.draw()
            win.flip()
            
            # Allow escape key to exit immediately
            if 'escape' in event.getKeys():
                win.close()
                core.quit()

    # After all characters are displayed, show the prompt and wait for user response
    prompt_text.draw()
    win.flip()
    keys = event.waitKeys(keyList=['left', 'right', 'escape'])
    if keys:
        if 'escape' in keys:
            win.close()
            core.quit()
        user_response = keys[0]
        print("User response:", user_response)
        if user_response == "right":
            user_response_list.append(1)
        elif user_response == "left":
            user_response_list.append(0)
    print(label, user_response_list[-1])



# Clean up: close the window and quit
win.close()
core.quit()




# from psychopy import visual, core, event
# import numpy as np

# # Set up the window
# win = visual.Window(size=[800, 600], color="black", units="norm")

# # Define the word to display and flicker parameters
# word = "table"
# flicker_freq = 12  # in Hz
# base_duration = 0.5  # seconds per character
# extra_duration = 0.2  # additional seconds for the last letter
# frame_rate = 60  # assumed refresh rate

# # Create a square frame stimulus that remains constant
# square_frame = visual.Rect(
#     win,
#     width=0.4,  # adjust size as needed
#     height=0.4,
#     pos=(0, 0),
#     lineColor="white",
#     fillColor=None,
#     lineWidth=2
# )

# # Prepare a text stimulus for displaying the letter.
# text_stim = visual.TextStim(
#     win,
#     text="",
#     pos=(0, 0),
#     color="white",
#     height=0.2
# )

# # Iterate through each character in the word.
# for i, letter in enumerate(word):
#     # Determine duration: add extra time for the last letter.
#     if i == len(word) - 1:
#         duration = base_duration + extra_duration
#     else:
#         duration = base_duration
    
#     clock = core.Clock()
    
#     # Display the letter with 12Hz flickering for the specified duration.
#     while clock.getTime() < duration:
#         t = clock.getTime()
#         # Flicker logic: letter is visible when the sine wave is positive.
#         if np.sin(2 * np.pi * flicker_freq * t) > 0:
#             text_stim.text = letter
#         else:
#             text_stim.text = ""
        
#         square_frame.draw()
#         text_stim.draw()
#         win.flip()
        
#         # Allow escape key to exit early.
#         if 'escape' in event.getKeys():
#             win.close()
#             core.quit()

# # Clean up: close the window and quit
# win.close()
# core.quit()
