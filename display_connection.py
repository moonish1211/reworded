# This is chatgpt output of integrating OpenBCI connection as well as data saving mechanism


## This code stores data into csv as this column
# iteration, label, user_input, chanel_0, chanel_1, ....., 
# Problem, the csv is skipping iterations --> Not fixed....

from psychopy import visual, core, event
import numpy as np
import random
import csv
import os
import json
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

## Preparing the data:
'''
Collecting 450 words to be tested on and the ratio of correct vs incorrect will be a parameter.
'''
correct_word_num = 250
incorrect_advanced_num = 100
incorrect_trick_num = 100

## Randomly get subset of these words from the txt file.
def select_random_words(file_path, num_words=250):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file if line.strip()]
    if len(words) < num_words:
        raise ValueError("The file contains fewer words than requested.")
    random_words = random.sample(words, num_words)
    return random_words

correct_word_path = "correct.txt" 
correct_words = select_random_words(correct_word_path, num_words=correct_word_num)
correct_words = [(word, 1) for word in correct_words]

incorrect_advanced_path = "incorrect_advance.txt"  
incorrect_advanced_words = select_random_words(incorrect_advanced_path, num_words=incorrect_advanced_num)
incorrect_advanced_words = [(word, 0) for word in incorrect_advanced_words]

incorrect_trick_path = "incorrect_trick.txt"  
incorrect_trick_words = select_random_words(incorrect_trick_path, num_words=incorrect_trick_num)
incorrect_trick_words = [(word, 0) for word in incorrect_trick_words]

# Combine and shuffle all lists
combined_list = correct_words + incorrect_advanced_words + incorrect_trick_words
random.shuffle(combined_list)

length_data = len(combined_list)

##############################
# Set up BrainFlow connection #
##############################
sampling_rate = 250  # Typical for the Cyton board
params = BrainFlowInputParams()
params.serial_port = "COM3"  # Update this port for your system (e.g., '/dev/ttyUSB0' on Linux)
board_id = 0  # Standard Cyton board ID
# board = BoardShim(board_id, params)

# # ##For now we are using synthetic board
board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
board.prepare_session()
board.start_stream(45000)

# Initialize previous sample count to segment data for each iteration.
previous_sample_count = board.get_board_data_count()

# CSV filename to store EEG data.
csv_filename = "openbci_data.csv"
# If the file doesn't exist, write the header.
header_written = os.path.exists(csv_filename)
csv_file = open(csv_filename, 'a', newline='')
csv_writer = csv.writer(csv_file)

##############################
# Set up display window (full screen)
##############################
# win = visual.Window(fullscr=True, color="black", units="norm")
win = visual.Window(size=[800, 600], color="black", units="norm")

# Display parameters
flicker_freq = 12          # Flicker frequency in Hz
base_duration = 0.5        # Duration per character in seconds
extra_duration = 0.2       # Extra duration for the last character

# Create a constant square frame stimulus
square_frame = visual.Rect(
    win,
    width=0.4,  # Adjust as needed
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
    width=0.1,   # Adjust size as needed
    height=0.1,
    pos=(-0.95, -0.95),  # Bottom left in normalized coordinates
    fillColor="white",
    lineColor="white"
)

##############################
# Write CSV header if needed
##############################
# We'll assume the board returns a fixed number of channels.
# For example, if using the Cyton board, you might have 8 EEG channels.
# Adjust n_channels to your configuration.
n_channels = 8
if not header_written:
    header = ['iteration','word','label', 'user_input'] + [f"channel_{i}" for i in range(n_channels)]
    csv_writer.writerow(header)
    csv_file.flush()

##############################
# Main Loop: Display words and record EEG data
##############################
label_list = []
user_response_list = []
word_shown = 0

for index in range(length_data):
    # print(i)
    word = combined_list[index][0]
    label= combined_list[index][1]
    label_list.append(label)
    
    # Display each character in the word sequentially
    for i, letter in enumerate(word):
        # Determine duration (last letter gets extra time)
        duration = base_duration + extra_duration if i == len(word) - 1 else base_duration
        clock = core.Clock()
        while clock.getTime() < duration:
            t = clock.getTime()
            # Flicker logic: here, we simply show the letter.
            text_stim.text = letter
            square_frame.draw()
            text_stim.draw()
            # Time-lock: flash the marker at the very start.
            if clock.getTime() < 0.02:
                flash_marker.draw()
            win.flip()
            if 'escape' in event.getKeys():
                win.close()
                board.stop_stream()
                board.release_session()
                csv_file.close()
                core.quit()
    
    # After displaying the word, show the prompt for 1.3 seconds and capture a response.
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
            board.stop_stream()
            board.release_session()
            csv_file.close()
            core.quit()
    if key_pressed:
        user_response_list.append(1)
    else:
        user_response_list.append(0)
    print(word, label, user_response_list[-1])
    word_shown += 1

    # Record EEG data for this iteration as one row.
    current_count = board.get_board_data_count()

    
    all_data = board.get_board_data()  # Shape: (channels, total_samples)
    iteration_data = all_data[:, previous_sample_count:current_count]
    previous_sample_count = current_count

    # For each channel, convert the data array to a list and then to a JSON string.
    row =  [index, word ,label, user_response_list[-1]]
    for ch in range(n_channels):
        # If the board returns more channels than we expect, limit to n_channels.
        channel_data = iteration_data[ch, :].tolist() if ch < iteration_data.shape[0] else []
        # Store as JSON string to preserve list structure.
        row.append(json.dumps(channel_data))
    csv_writer.writerow(row)
    csv_file.flush()


    # if current_count > previous_sample_count:
    #     # Get the new data collected from the board.
    #     all_data = board.get_board_data()  # Shape: (channels, total_samples)
    #     iteration_data = all_data[:, previous_sample_count:current_count]
    #     previous_sample_count = current_count
        
    #     # For each channel, convert the data array to a list and then to a JSON string.
    #     row =  [index, word ,label, user_response_list[-1]]
    #     for ch in range(n_channels):
    #         # If the board returns more channels than we expect, limit to n_channels.
    #         channel_data = iteration_data[ch, :].tolist() if ch < iteration_data.shape[0] else []
    #         # Store as JSON string to preserve list structure.
    #         row.append(json.dumps(channel_data))
    #     csv_writer.writerow(row)
    #     csv_file.flush()
    
    # Every 25 words, give the user a break.
    if word_shown % 25 == 0:
        while True:
            prompt_break.draw()
            win.flip()
            keys = event.getKeys()
            if 'space' in keys:
                print("Spacebar pressed! Resuming experiment...")
                break
            core.wait(0.1)

##############################
# Clean up: Stop board, close CSV, and close window.
##############################
board.stop_stream()
board.release_session()
csv_file.close()
win.close()
core.quit()





# ## This is dumping the data to csv after experiment, but coming across same issue still... 

# from psychopy import visual, core, event
# import numpy as np
# import random
# import csv
# import os
# import json
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ## Preparing the data:
# '''
# Collecting 500 words to be tested on and the ratio of correct vs incorrect will be a parameter.
# '''
# correct_word_num = 250
# incorrect_advanced_num = 150
# incorrect_trick_num = 100

# ## Randomly get subset of these words from the txt file.
# def select_random_words(file_path, num_words=250):
#     with open(file_path, 'r') as file:
#         words = [line.strip() for line in file if line.strip()]
#     if len(words) < num_words:
#         raise ValueError("The file contains fewer words than requested.")
#     random_words = random.sample(words, num_words)
#     return random_words

# correct_word_path = "correct.txt" 
# correct_words = select_random_words(correct_word_path, num_words=correct_word_num)
# correct_words = [(word, 1) for word in correct_words]

# incorrect_advanced_path = "incorrect_advance.txt"  
# incorrect_advanced_words = select_random_words(incorrect_advanced_path, num_words=incorrect_advanced_num)
# incorrect_advanced_words = [(word, 0) for word in incorrect_advanced_words]

# incorrect_trick_path = "incorrect_trick.txt"  
# incorrect_trick_words = select_random_words(incorrect_trick_path, num_words=incorrect_trick_num)
# incorrect_trick_words = [(word, 0) for word in incorrect_trick_words]

# # Combine and shuffle all lists
# combined_list = correct_words + incorrect_advanced_words + incorrect_trick_words
# random.shuffle(combined_list)

# ##############################
# # Set up BrainFlow connection #
# ##############################
# sampling_rate = 250  # Typical for the Cyton board
# params = BrainFlowInputParams()
# params.serial_port = "COM3"  # Update this port for your system (e.g., '/dev/ttyUSB0' on Linux)
# board_id = 0  # Standard Cyton board ID
# # board = BoardShim(board_id, params)

# # # # ##For now we are using synthetic board
# board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
# board.prepare_session()
# board.start_stream(45000)

# # Initialize previous sample count to segment data for each iteration.
# previous_sample_count = board.get_board_data_count()


# ##############################
# # Set up display window (full screen)
# ##############################
# win = visual.Window(fullscr=True, color="black", units="norm")

# # Display parameters
# flicker_freq = 12          # Flicker frequency in Hz
# base_duration = 0.5        # Duration per character in seconds
# extra_duration = 0.2       # Extra duration for the last character

# # Create a constant square frame stimulus
# square_frame = visual.Rect(
#     win,
#     width=0.4,  # Adjust as needed
#     height=0.4,
#     pos=(0, 0),
#     lineColor="white",
#     fillColor=None,
#     lineWidth=2
# )

# # Prepare a text stimulus for displaying letters
# text_stim = visual.TextStim(
#     win,
#     text="",
#     pos=(0, 0),
#     color="white",
#     height=0.2
# )

# # Prepare prompt screens
# prompt_text = visual.TextStim(
#     win,
#     text="Click any key if you recognized this as vocabulary!!",
#     pos=(0, 0),
#     color="white",
#     height=0.07
# )
# prompt_break = visual.TextStim(
#     win,
#     text="Break time!! Press space to resume the experiment",
#     pos=(0, 0),
#     color="white",
#     height=0.07
# )

# # Create a flash marker stimulus at the bottom left for time-locking
# flash_marker = visual.Rect(
#     win,
#     width=0.1,   # Adjust size as needed
#     height=0.1,
#     pos=(-0.95, -0.95),  # Bottom left in normalized coordinates
#     fillColor="white",
#     lineColor="white"
# )

# ##############################
# # Prepare to accumulate data rows in a list.
# ##############################
# data_rows = []  
# # We'll assume the board returns a fixed number of channels.
# n_channels = 8  # Adjust to your board's configuration

# ##############################
# # Main Loop: Display words and record EEG data
# ##############################
# label_list = []
# user_response_list = []
# word_shown = 0

# for word, label in combined_list:
#     label_list.append(label)
    
#     # Display each character in the word sequentially
#     for i, letter in enumerate(word):
#         # Determine the display duration: last letter gets extra time
#         duration = base_duration + extra_duration if i == len(word) - 1 else base_duration
#         clock = core.Clock()
#         while clock.getTime() < duration:
#             t = clock.getTime()
#             # For simplicity, we show the letter continuously (modify for flicker if desired)
#             text_stim.text = letter
#             square_frame.draw()
#             text_stim.draw()
#             # Time-lock: flash the marker at the very start.
#             if clock.getTime() < 0.02:
#                 flash_marker.draw()
#             win.flip()
#             if 'escape' in event.getKeys():
#                 win.close()
#                 board.stop_stream()
#                 board.release_session()
#                 core.quit()
    
#     # After displaying the word, show the prompt for 1.3 seconds and capture a response.
#     user_response = None
#     key_pressed = False
#     clock = core.Clock()
#     while clock.getTime() < 1.3:
#         prompt_text.draw()
#         win.flip()
#         keys = event.getKeys()
#         if keys and user_response is None:
#             user_response = keys[0]  # Record the first key press
#             key_pressed = True
#         if 'escape' in keys:
#             win.close()
#             board.stop_stream()
#             board.release_session()
#             core.quit()
#     if key_pressed:
#         user_input = 1
#         user_response_list.append(1)
#     else:
#         user_input = 0
#         user_response_list.append(0)
#     print(label, user_input)
#     word_shown += 1

#     # Record EEG data for this iteration.
#     current_count = board.get_board_data_count()
#     if current_count > previous_sample_count:
#         # Get the new data collected from the board.
#         all_data = board.get_board_data()  # Shape: (channels, total_samples)
#         iteration_data = all_data[:, previous_sample_count:current_count]
#         previous_sample_count = current_count
        
#         # For each channel, convert the channel data to a list
#         row = [word_shown, label, user_input]
#         for ch in range(n_channels):
#             channel_data = iteration_data[ch, :].tolist() if ch < iteration_data.shape[0] else []
#             # Append the JSON string representation of the channel's data
#             row.append(json.dumps(channel_data))
#         data_rows.append(row)
    
#     # Every 25 words, give the user a break.
#     if word_shown % 25 == 0:
#         while True:
#             prompt_break.draw()
#             win.flip()
#             keys = event.getKeys()
#             if 'space' in keys:
#                 print("Spacebar pressed! Resuming experiment...")
#                 break
#             core.wait(0.1)

# ##############################
# # At the end of the experiment, dump all data rows to CSV.
# ##############################
# csv_filename = "openbci_data.csv"
# with open(csv_filename, 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     # Write header
#     header = ['iteration', 'label', 'user_input'] + [f"channel_{i}" for i in range(n_channels)]
#     csv_writer.writerow(header)
#     # Write all rows
#     csv_writer.writerows(data_rows)

# ##############################
# # Clean up: Stop board and close window.
# ##############################
# board.stop_stream()
# board.release_session()
# win.close()
# core.quit()


