from psychopy import visual, core, event
import numpy as np
import random
from brainflow import BrainFlowInputParams

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
## Code for doing it in pop up to look at terminal output
# win = visual.Window(size=[800, 600], color="black", units="norm")
## Code for fullscreen
win = visual.Window(fullscr=True, color="black", units="norm")

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
    text="Click any key if you recognized this as vocabulary!!",
    pos=(0, 0),
    color="white",
    height=0.07
)
prompt_break = visual.TextStim(
    win,
    text="Break time!! Press space to resume the experiment",
    pos=(0, 0),
    color="white",
    height=0.07
)

# Create a flash marker stimulus at the bottom left.
# This marker will be drawn (flashed) to indicate event timing.
flash_marker = visual.Rect(
    win,
    width=0.1,   # Adjust size as needed
    height=0.1,
    pos=(-0.95, -0.95),  # Bottom left in normalized coordinates
    fillColor="white",
    lineColor="white"
)


label_list = []
user_response_list = []
word_shown = 0
for word, label in combined_list:

    # Initialize flags to ensure the flash is drawn only once at each desired time
    # flash_shown_first = False
    # flash_shown_last = False


    label_list.append(label)
    # Display each character in the word sequentially
    for i, letter in enumerate(word):
        # Determine the display duration: last letter gets extra time
        duration = base_duration + extra_duration if i == len(word) - 1 else base_duration
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime()
            # Flicker logic: show the letter when the sine wave is positive, otherwise blank
            text_stim.text = letter
            # if np.sin(2 * np.pi * flicker_freq * t) > 0:
            #     text_stim.text = letter
            # else:
            #     text_stim.text = ""
            
            square_frame.draw()
            text_stim.draw()


            # --- Time Locking via Flash Marker ---
            # At the very start of the first character, flash the marker.
            # if i == 0 and not flash_shown_first and clock.getTime() < 0.02:
            if clock.getTime() < 0.02:
                flash_marker.draw()
                flash_shown_first = True
            # At the very end of the last character display, flash the marker.
            # if i == len(word) - 1 and not flash_shown_last and clock.getTime() > duration - 0.02:
            #     flash_marker.draw()
            #     flash_shown_last = True
            # # -------------------------------------



            win.flip()
            
            # Allow escape key to exit immediately
            if 'escape' in event.getKeys():
                win.close()
                core.quit()

    # After displaying the word, show the prompt screen for 1.3 seconds
    user_response = None
    key_pressed = False
    clock = core.Clock()
    while clock.getTime() < 1.3:
        prompt_text.draw()
        win.flip()
        keys = event.getKeys()  # Check for any key press
        if keys and user_response is None:
            user_response = keys[0]  # Record the first key press
            key_pressed = True
        if 'escape' in keys:
            win.close()
            core.quit()
    if key_pressed:
        user_response_list.append(1)
    else:
        user_response_list.append(0)
    print(label, user_response_list[-1])
    word_shown += 1

    if word_shown % 25 == 0:
        ##For every 25 word, we would like to give break to the user that pauses the loop until key is pressed. 
        while True:
            prompt_break.draw()
            win.flip()
            keys = event.getKeys()  # Check for key presses
            if 'space' in keys:  # If spacebar is pressed
                print("Spacebar pressed! Exiting loop...")
                break
            core.wait(0.1)  # Small delay to prevent high CPU usage
            ### This may be a bad lag for implementation...


# Clean up: close the window and quit
win.close()
core.quit()
