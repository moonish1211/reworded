# This script runs a visually evoked potential (VEP)-based experiment.
# The BCI/EEG acquisition part is disabled (cyton_in is set to False) so that you can focus on the display.
# Calibration mode is enabled so that the display loop runs.

from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle
import mne

# -----------------------
# 0. Data Aquizition 
# -----------------------

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



# -------------------------------
# 1. Experiment Setup and Configuration
# -------------------------------
cyton_in = False         # Disable EEG data acquisition for display testing
lsl_out = False
width = 1536
height = 864
aspect_ratio = width/height
refresh_rate = 60.02
stim_duration = 1.2
n_per_class = 2
stim_type = 'alternating'  # 'alternating' or 'independent'
subject = 1
session = 1
calibration_mode = True   # Enable calibration mode so the display trial loop runs


#### FIX
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

import string
import numpy as np
import psychopy.visual
import psychopy.event
from psychopy import core

# -------------------------------
# 2. Visual Stimuli Preparation
# -------------------------------

letters = 'QAZ⤒WSX,EDC?R⌫FVT⎵GBYHN.UJMPIKOL'
win = psychopy.visual.Window(
        size=(800, 800),
        units="norm",
        fullscr=False)
n_text = 32
text_cap_size = 64
text_strip_height = n_text * text_cap_size
text_strip = np.full((text_strip_height, text_cap_size), np.nan)
text = psychopy.visual.TextStim(win=win, height=0.145, font="Helvetica")
cap_rect_norm = [-(text_cap_size / 2.0) / (win.size[0] / 2.0),
                 +(text_cap_size / 2.0) / (win.size[1] / 2.0),
                 +(text_cap_size / 2.0) / (win.size[0] / 2.0),
                 -(text_cap_size / 2.0) / (win.size[1] / 2.0)]
for (i_letter, letter) in enumerate(letters):
    text.text = letter.upper()
    buff = psychopy.visual.BufferImageStim(
        win=win,
        stim=[text],
        rect=cap_rect_norm)
    i_rows = slice(i_letter * text_cap_size,
                    i_letter * text_cap_size + text_cap_size)
    text_strip[i_rows, :] = (np.flipud(np.array(buff.image)[..., 0]) / 255.0 * 2.0 - 1.0)
new_size = max([int(np.power(2, np.ceil(np.log(dim_size) / np.log(2))))
                for dim_size in text_strip.shape])
pad_amounts = []
for i_dim in range(2):
    first_offset = int((new_size - text_strip.shape[i_dim]) / 2.0)
    second_offset = new_size - text_strip.shape[i_dim] - first_offset
    pad_amounts.append([first_offset, second_offset])
text_strip = np.pad(
    array=text_strip,
    pad_width=pad_amounts,
    mode="constant",
    constant_values=0.0)
text_strip = (text_strip - 1) * -1
el_mask = np.ones(text_strip.shape) * -1.0
el_mask[:text_cap_size, :text_cap_size] = 1.0
el_mask = np.roll(el_mask,
                    (int(new_size / 2 - text_cap_size / 2),) * 2,
                    axis=(0, 1))
base_phase = ((text_cap_size * (n_text / 2.0)) - (text_cap_size / 2.0)) / new_size
phase_inc = (text_cap_size) / float(new_size)
phases = np.array([
    (0.0, base_phase - i_letter * phase_inc)
    for i_letter in range(n_text)])
win.close()
print(text_strip.shape, el_mask.shape, phases.shape)

def create_32_targets(size=2/8*0.7, colors=[-1, -1, -1] * 32, checkered=False, elementTex=None, elementMask=None, phases=None):
    width, height = window.size
    aspect_ratio = width/height
    positions = create_32_target_positions(size)
    if checkered:
        texture = checkered_texure()
    else:
        texture = elementTex
    keys = visual.ElementArrayStim(window, nElements=32, elementTex=texture, elementMask=elementMask, units='norm',
                                   sizes=[size, size * aspect_ratio], xys=positions, phases=phases, colors=colors)
    return keys

def create_32_key_caps(size=2/8*0.7, colors=[-1, -1, -1] * 32):
    width, height = window.size
    aspect_ratio = width/height
    positions = create_32_target_positions(size)
    positions = [[pos[0]*width/2, pos[1]*height/2] for pos in positions]
    keys = visual.ElementArrayStim(window, nElements=32, elementTex=text_strip, elementMask=el_mask, units='pix',
                                   sizes=text_strip.shape, xys=positions, phases=phases, colors=colors)
    return keys

def checkered_texure():
    rows = 8
    cols = 8
    array = np.zeros((rows, cols))
    for i in range(rows):
        array[i, ::2] = i % 2
        array[i, 1::2] = (i+1) % 2
    return np.kron(array, np.ones((16, 16)))*2-1

def create_32_target_positions(size=2/8*0.7):
    size_with_border = size / 0.7
    width, height = window.size
    aspect_ratio = width/height
    positions = []
    for i_col in range(8):
        positions.extend([[i_col*size_with_border-1+size_with_border/2, -j_row*size_with_border*aspect_ratio+1-size_with_border*aspect_ratio/2 - 1/4/2] for j_row in range(4)])
    return positions

def create_photosensor_dot(size=2/8*0.7):
    width, height = window.size
    ratio = width/height
    return visual.Rect(win=window, units="norm", width=size, height=size * ratio, 
                       fillColor='white', lineWidth=0, pos=[1 - size/2, -1 - size/8]
    )

def create_trial_sequence(n_per_class, classes=[(7.5, 0), (8.57, 0), (10, 0), (12, 0), (15, 0)], seed=0):
    seq = classes * n_per_class
    random.seed(seed)
    random.shuffle(seq)
    return seq

# -------------------------------
# Create the Main PsychoPy Window and Stimuli
# -------------------------------
keyboard = keyboard.Keyboard()
window = visual.Window(
        size=[width, height],
        checkTiming=True,
        allowGUI=False,
        fullscr=True,
        useRetina=False,
    )
visual_stimulus = create_32_targets(checkered=False)
key_caps = create_32_key_caps()
photosensor_dot = create_physical_dot = create_photosensor_dot()
photosensor_dot.color = np.array([-1, -1, -1])
photosensor_dot.draw()
window.flip()

# -------------------------------
# 3. (Skipped) EEG Acquisition Setup
# -------------------------------
# The following block is skipped because cyton_in is False.
if cyton_in:
    # (Original code for connecting to the Cyton board and starting data acquisition)
    pass

# -------------------------------
# 4. Stimulus Presentation and Trial Sequence Setup
# -------------------------------
num_frames = np.round(stim_duration * refresh_rate).astype(int)
frame_indices = np.arange(num_frames)
if stim_type == 'alternating':
    stimulus_classes = [(8, 0), (8, 0.5), (8, 1), (8, 1.5),
                        (9, 0), (9, 0.5), (9, 1), (9, 1.5),
                        (10, 0), (10, 0.5), (10, 1), (10, 1.5),
                        (11, 0), (11, 0.5), (11, 1), (11, 1.5),
                        (12, 0), (12, 0.5), (12, 1), (12, 1.5),
                        (13, 0), (13, 0.5), (13, 1), (13, 1.5),
                        (14, 0), (14, 0.5), (14, 1), (14, 1.5),
                        (15, 0), (15, 0.5), (15, 1), (15, 1.5), ]
    stimulus_frames = np.zeros((num_frames, len(stimulus_classes)))
    for i_class, (flickering_freq, phase_offset) in enumerate(stimulus_classes):
            phase_offset += .00001
            stimulus_frames[:, i_class] = signal.square(2 * np.pi * flickering_freq * (frame_indices / refresh_rate) + phase_offset * np.pi)
trial_sequence = np.tile(np.arange(32), n_per_class)
np.random.seed(run)
np.random.shuffle(trial_sequence)
target_positions = create_32_target_positions(size=2/8*0.7)

# Initialize arrays for simulation purposes (these won't be used when cyton_in is False)
eeg = np.zeros((8, 0))
aux = np.zeros((3, 0))
timestamp = np.zeros((0))
eeg_trials = []
aux_trials = []
trial_ends = []
skip_count = 0
accuracy = 0
predictions = []
aim_target_color = 'white'

# -------------------------------
# 5. Calibration Mode: Stimulus Presentation and (Simulated) Trial Processing
# -------------------------------
if calibration_mode:
    for i_trial, target_id in enumerate(trial_sequence):
        print(f"Trial {i_trial+1}, target: {target_id}")
        trial_text = visual.TextStim(window, text=f'Trial {i_trial+1}/{len(trial_sequence)}', pos=(0, -0.93), color='white', units='norm', height=0.07)
        trial_text.draw()
        accuracy_text = visual.TextStim(window, text=f'Accuracy: {accuracy*100:.2f}%', pos=(0, 0.93), color=aim_target_color, units='norm', height=0.07)
        accuracy_text.draw()
        visual_stimulus.colors = np.array([-1] * 3).T
        visual_stimulus.draw()
        key_caps.draw()
        photosensor_dot.color = np.array([-1, -1, -1])
        photosensor_dot.draw()
        aim_target = visual.Rect(win=window, units="norm", width=2/8*0.7 * 1.3, height=2/8*0.7*aspect_ratio * 1.3, pos=target_positions[target_id], lineColor=aim_target_color, lineWidth=3)
        aim_target.draw()
        window.flip()
        core.wait(0.7)
        finished_displaying = False
        while not finished_displaying:
            for i_frame in range(num_frames):
                next_flip = window.getFutureFlipTime()
                keys = keyboard.getKeys()
                if 'escape' in keys:
                    core.quit()
                visual_stimulus.colors = np.array([stimulus_frames[i_frame]] * 3).T
                visual_stimulus.draw()
                photosensor_dot.color = np.array([1, 1, 1])
                photosensor_dot.draw()
                trial_text.draw()
                aim_target.draw()
                if core.getTime() > next_flip and i_frame != 0:
                    print('Missed frame')
                window.flip()
            finished_displaying = True
        # Since BCI data acquisition is disabled, simulate a brief pause after each trial
        core.wait(1)
        
    # No data saving or board shutdown is necessary when cyton_in is False.
