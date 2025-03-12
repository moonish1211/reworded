from psychopy import visual, core, event
import numpy as np
import random
import csv
import os
import json
from brainflow import BrainFlowInputParams, BoardShim, BoardIds
from datetime import datetime
import sys, glob, serial, time
from serial import Serial


## Preparing the data:
correct_word_num = 250
incorrect_advanced_num = 100
incorrect_trick_num = 100

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

# Combine and shuffle the word lists
combined_list = correct_words + incorrect_advanced_words + incorrect_trick_words
random.shuffle(combined_list)
length_data = len(combined_list)


def find_openbci_port():
    """Finds the port to which the Cyton Dongle is connected to."""
    # Find serial port names per OS
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass
    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
        exit()
    else:
        return openbci_port


##############################
# Set up BrainFlow connection #
##############################
sampling_rate = 250  # Typical for the Cyton board
params = BrainFlowInputParams()
# params.serial_port = "COM3"  # Update as needed for your system
# board_id =   # Standard Cyton board ID
# board = BoardShim(board_id, params)

# # board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
CYTON_BOARD_ID = 0
BAUD_RATE = 115200
if CYTON_BOARD_ID != 6:
    params.serial_port = find_openbci_port()
elif CYTON_BOARD_ID == 6:
    params.ip_port = 9000
board = BoardShim(CYTON_BOARD_ID, params)
board.prepare_session()
board.start_stream(45000)

# Initialize a variable to track the board's sample count
previous_sample_count = board.get_board_data_count()

now = datetime.now()
timestamp = now.strftime("%m-%d_%H-%M")

# Set up the window (fullscreen)
win = visual.Window(fullscr=True, color="black", units="norm")

# Define display parameters
flicker_freq = 12          # Flicker frequency in Hz (unused in this version)
base_duration = 0.5        # Duration per character in seconds
extra_duration = 0.2       # Extra duration for the last character

# Create stimuli: square frame, text stimulus, prompt, break, and flash marker
square_frame = visual.Rect(
    win,
    width=0.4,
    height=0.4,
    pos=(0, 0),
    lineColor="white",
    fillColor=None,
    lineWidth=2
)

text_stim = visual.TextStim(
    win,
    text="",
    pos=(0, 0),
    color="white",
    height=0.2
)

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

flash_marker = visual.Rect(
    win,
    width=0.1,
    height=0.1,
    pos=(0.95, -0.95),
    fillColor="white",
    lineColor="white"
)

n_channels = 8
csv_filename = f"data/experiment_data_{timestamp}.csv"
csv_headers = ['word','label', 'user_input'] + [f"channel_{i}" for i in range(n_channels)]

# Function to delete the last row from a CSV file (if needed)
def delete_last_row(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) > 1:
        lines = lines[:-1]
    with open(filename, 'w', newline='') as f:
        f.writelines(lines)

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)

    user_response_list = []
    word_shown = 0

    for index in range(length_data):
        word = combined_list[index][0]
        label = combined_list[index][1]

        iteration_data = None  # To store EEG data from the last letter display

        # Loop through each character in the word
        for i, letter in enumerate(word):
            # For the last character, collect EEG data during its display
            if i == len(word) - 1:
                # Record starting sample count for EEG data collection
                last_letter_start = board.get_board_data_count()
                duration = base_duration + extra_duration
                clock = core.Clock()
                while clock.getTime() < duration:
                    text_stim.text = letter
                    square_frame.draw()
                    text_stim.draw()
                    # Flash marker at the start of the display
                    if clock.getTime() < 0.02:
                        flash_marker.draw()
                    win.flip()
                    if 'escape' in event.getKeys():
                        win.close()
                        board.stop_stream()
                        board.release_session()
                        core.quit()
                # Record ending sample count and extract data for the last letter
                current_count = board.get_board_data_count()
                iteration_data = board.get_board_data()[:, last_letter_start:current_count]
                # Update previous_sample_count to avoid overlapping data in future iterations
                previous_sample_count = current_count
            else:
                # For non-last letters, just display without EEG data collection
                duration = base_duration
                clock = core.Clock()
                while clock.getTime() < duration:
                    text_stim.text = letter
                    square_frame.draw()
                    text_stim.draw()
                    if clock.getTime() < 0.02:
                        flash_marker.draw()
                    win.flip()
                    if 'escape' in event.getKeys():
                        win.close()
                        board.stop_stream()
                        board.release_session()
                        core.quit()

        # After word display, show the prompt screen (this part now does not collect EEG data)
        user_response = None
        key_pressed = False
        clock = core.Clock()
        while clock.getTime() < 1.3:
            prompt_text.draw()
            win.flip()
            keys = event.getKeys()
            if keys and user_response is None:
                user_response = keys[0]
                key_pressed = True
            if 'escape' in keys:
                win.close()
                board.stop_stream()
                board.release_session()
                core.quit()

        # In case the EEG data segment was not captured properly, wait until data are available
        while iteration_data is None or iteration_data.shape[1] == 0:
            print(f"Waiting for EEG data for {word}...")
            current_count = board.get_board_data_count()
            iteration_data = board.get_board_data()[:, previous_sample_count:current_count]
            previous_sample_count = current_count
            core.wait(0.05)

        if key_pressed:
            user_response_list.append(1)
        else:
            user_response_list.append(0)
        print(word, label, user_response_list[-1])

        # Write the results along with EEG data to CSV
        row = [word, label, user_response_list[-1]]
        for ch in range(n_channels):
            channel_data = iteration_data[ch, :].tolist() if ch < iteration_data.shape[0] else []
            row.append(json.dumps(channel_data))
        writer.writerow(row)

        word_shown += 1

        # Every 25 words, allow a break until space is pressed.
        if word_shown % 25 == 0:
            while True:
                prompt_break.draw()
                win.flip()
                keys = event.getKeys()
                if 'space' in keys:
                    print("Spacebar pressed! Exiting break...")
                    break
                core.wait(0.1)

# Clean up: close the window and quit
win.close()
core.quit()
