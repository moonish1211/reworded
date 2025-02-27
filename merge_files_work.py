# This script runs a visually evoked potential (VEP)-based experiment.
# The BCI/EEG acquisition part is disabled (cyton_in is set to False) so that you can focus on the display.
# Calibration mode is enabled so that the display trial loop runs.
#
# In this modified version, the display/trial loop from displays.py has been integrated to replace
# the original 32-target code. The BCI/EEG parts (and associated variables) remain intact.

from psychopy import visual, core, event  # note: event is now imported for the display loop
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle
import mne
import string

# -----------------------
# 0. Data Acquisition 
# -----------------------

# Define number of words for each category
correct_word_num = 250
incorrect_advanced_num = 150
incorrect_trick_num = 100

# Randomly get subset of words from a text file
def select_random_words(file_path, num_words=250):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file if line.strip()]
    if len(words) < num_words:
        raise ValueError("The file contains fewer words than requested.")
    return random.sample(words, num_words)

correct_word_path = "correct.txt" 
correct_words = select_random_words(correct_word_path, num_words=correct_word_num)
correct_words = [(word, 1) for word in correct_words]

incorrect_advanced_path = "incorrect_advance.txt"  
incorrect_advanced_words = select_random_words(incorrect_advanced_path, num_words=incorrect_advanced_num)
incorrect_advanced_words = [(word, 0) for word in incorrect_advanced_words]

incorrect_trick_path = "incorrect_trick.txt"  
incorrect_trick_words = select_random_words(incorrect_trick_path, num_words=incorrect_trick_num)
incorrect_trick_words = [(word, 0) for word in incorrect_trick_words]

# Combine and shuffle the lists
combined_list = correct_words + incorrect_advanced_words + incorrect_trick_words
random.shuffle(combined_list)

# -------------------------------
# 1. Experiment Setup and Configuration
# -------------------------------
cyton_in = False         # Disable EEG data acquisition for display testing
lsl_out = False
width = 1536
height = 864
aspect_ratio = width / height
refresh_rate = 60.02
stim_duration = 1.2
n_per_class = 2
stim_type = 'alternating'  # 'alternating' or 'independent'
subject = 1
session = 1
calibration_mode = True   # Enable calibration mode so that the display trial loop runs

# Directory and file paths (not used when cyton_in is False)
save_dir = f'data/cyton8_{stim_type}-vep_32-class_{stim_duration}s/sub-{subject:02d}/ses-{session:02d}/'
run = 1
save_file_eeg = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_timestamp = save_dir + f'timestamp_{n_per_class}-per-class_run-{run}.npy'
save_file_metadata = save_dir + f'metadata_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'
model_file_path = 'cache/FBTRCA_model.pkl'

# -------------------------------
# 2. (Skipped) EEG Acquisition Setup
# -------------------------------
# The following block is skipped because cyton_in is False.
if cyton_in:
    # (Original code for connecting to the Cyton board and starting data acquisition)
    pass

# -------------------------------
# 3. Display / Calibration Mode
# -------------------------------
# Instead of running the original 32-target stimulus loop, we now run the display code from displays.py.
# (The EEG-related code remains untouched above.)
if calibration_mode:
    # Create a new window for the display (as in displays.py)
    win = visual.Window(size=[800, 600], color="black", units="norm")
    
    # Display parameters
    flicker_freq = 12          # Flicker frequency in Hz
    base_duration = 0.5        # Duration per character in seconds
    extra_duration = 0.2       # Extra duration for the last character
    frame_rate = 60            # Assumed monitor refresh rate

    # Create a constant square frame stimulus
    square_frame = visual.Rect(
        win,
        width=0.4,
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
    
    # Prepare prompt screens
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
    
    # Create a flash marker stimulus at the bottom left for time-locking
    flash_marker = visual.Rect(
        win,
        width=0.05,
        height=0.05,
        pos=(-0.95, -0.95),
        fillColor="white",
        lineColor="white"
    )
    
    label_list = []
    user_response_list = []
    word_shown = 0

    # Loop over each word in the combined list
    for word, label in combined_list:
        # Initialize flags so that the flash marker is drawn only once per word at the beginning and end
        flash_shown_first = False
        flash_shown_last = False
        
        label_list.append(label)
        
        # Display each letter in the word sequentially
        for i, letter in enumerate(word):
            # Last letter gets extra display duration
            duration = base_duration + extra_duration if i == len(word) - 1 else base_duration
            clock = core.Clock()
            while clock.getTime() < duration:
                t = clock.getTime()
                # Flicker logic: display the letter when the sine wave is positive
                if np.sin(2 * np.pi * flicker_freq * t) > 0:
                    text_stim.text = letter
                else:
                    text_stim.text = ""
                
                # Draw the square frame and the letter
                square_frame.draw()
                text_stim.draw()
                
                # Flash marker at the very start of the first letter display
                if i == 0 and not flash_shown_first and clock.getTime() < 0.02:
                    flash_marker.draw()
                    flash_shown_first = True
                # Flash marker at the very end of the last letter display
                if i == len(word) - 1 and not flash_shown_last and clock.getTime() > duration - 0.02:
                    flash_marker.draw()
                    flash_shown_last = True
                    
                win.flip()
                
                # Allow exit if escape key is pressed
                if 'escape' in event.getKeys():
                    win.close()
                    core.quit()
                    
        # After displaying the word, show a prompt for a short duration (1.3 sec) to record user response
        user_response = None
        key_pressed = False
        clock = core.Clock()
        while clock.getTime() < 1.3:
            prompt_text.draw()
            win.flip()
            keys = event.getKeys()
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

        # Every 25 words, provide a break until the spacebar is pressed.
        if word_shown % 25 == 0:
            while True:
                prompt_break.draw()
                win.flip()
                keys = event.getKeys()
                if 'space' in keys:
                    print("Spacebar pressed! Exiting break...")
                    break
                core.wait(0.1)

    # Clean up: close the window and exit
    win.close()
    core.quit()

# -------------------------------
# (Any further code such as data saving would go here.)
