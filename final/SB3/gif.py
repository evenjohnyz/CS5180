import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
import imageio
from PIL import Image
import time

# set path
MODEL_PATH = "./models/sac_her_No_DQ_panda_pick_and_place.zip"
OUTPUT_VIDEO = "panda_PickAndPlace_demo_10s.mp4"

# create env
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")

# load model
model = SAC.load(MODEL_PATH, env=env)
print(f"Loaded from {MODEL_PATH} successfully")

frames = []

# set parameters
num_episodes = 30       
target_video_length = 10  

print("Startomg...")

for episode in range(num_episodes):
    print(f"Starting {episode+1}/{num_episodes} th grasping")
    
    # reset
    observation, _ = env.reset()
    
    # conduct current task
    steps = 0
    max_steps = 50  # maximum timestep 
    
    while steps < max_steps:
        steps += 1
        
        action, _ = model.predict(observation, deterministic=True)
        
        observation, reward, done, truncated, _ = env.step(action)
        
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        
        # if finish the current task, start the next
        if done or truncated:
            print(f"Compelete {episode+1}th grasping after  {steps} steps")
            break
    
    # if not finish in advance, finish it
    if not (done or truncated):
        print(f"{episode+1} grasping reaches time steps limit")
    
    if len(frames) > 1000:
        print(f"Already collect {len(frames)} framse，reaching the limit, stop recording")
        break

env.close()
print(f"\nCompelet recording {len(frames)} frames in total")


fps = len(frames) / target_video_length
print(f"Calculated frame rate: {fps:.1f} fps (used to compress the video to 10 seconds)")

# save to video
if len(frames) > 0:
    max_reasonable_fps = 60  
    
    if fps > max_reasonable_fps:
        # sample to get reasonable frame rate
        sampling_rate = int(fps / max_reasonable_fps)
        sampled_frames = frames[::sampling_rate]
        adjusted_fps = len(sampled_frames) / target_video_length
        
        print(f"The original frame rate is too high, sampled down from {len(frames)} frames to {len(sampled_frames)} frames")
        print(f"Adjusted frame rate: {adjusted_fps:.1f} fps")
        
        imageio.mimsave(
            OUTPUT_VIDEO, 
            [np.array(frame) for frame in sampled_frames], 
            fps=adjusted_fps,
            quality=9
        )
    else:
        # using original frame rate directly
        imageio.mimsave(
            OUTPUT_VIDEO, 
            [np.array(frame) for frame in frames], 
            fps=fps,
            quality=9
        )
    
    print(f"Video saved to {OUTPUT_VIDEO}")
    print(f"Video duration is approximately {target_video_length} seconds")
else:
    print("No frames captured, unable to create video") 