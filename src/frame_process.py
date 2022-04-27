import cv2
import numpy as np

def process(obs):
    '''
    Essentially we want to convert our atari game to grey and 84x84
    '''
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = obs[8:-12, 4:-12] #this crops our game
    obs = np.ascontiguousarray(obs, dtype=np.float32)/255   #scales to [0,1]

    return cv2.resize(obs, (84, 84), interpolation = cv2.INTER_AREA)

def stack_frames(frames, state, is_new=False):
    """
    For the game environment, we want to stack the four frames of the last four actions.
    """
    frame = process(state)
    
    if is_new == True:
        frames = np.stack(arrays=[frame, frame, frame, frame])
    else:
        frames[0] = frames[1]
        frames[1] = frames[2]
        frames[2] = frames[3]
        frames[3] = frame
    return frames
    
    
