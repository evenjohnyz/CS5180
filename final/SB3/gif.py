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
            print(f"  第 {episode+1} 次抓取在 {steps} 步后完成")
            break
    
    # if not finish in advance, finish it
    if not (done or truncated):
        print(f"  第 {episode+1} 次抓取达到步数上限")
    
    if len(frames) > 1000:
        print(f"已收集 {len(frames)} 帧，达到上限，停止录制")
        break

env.close()
print(f"\n录制完成，共 {len(frames)} 帧")

# 计算适当的帧率以达到目标视频长度
fps = len(frames) / target_video_length
print(f"计算的帧率: {fps:.1f} fps (用于将视频压缩至10秒)")

# 保存为视频
if len(frames) > 0:
    # 如果帧率过高，考虑抽样帧以降低帧率
    max_reasonable_fps = 60  # 大多数视频播放器能良好支持的最高帧率
    
    if fps > max_reasonable_fps:
        # 需要抽样以获得合理的帧率
        sampling_rate = int(fps / max_reasonable_fps)
        sampled_frames = frames[::sampling_rate]
        adjusted_fps = len(sampled_frames) / target_video_length
        
        print(f"原始帧率太高，从 {len(frames)} 帧中抽样为 {len(sampled_frames)} 帧")
        print(f"调整后的帧率: {adjusted_fps:.1f} fps")
        
        # 使用抽样后的帧
        imageio.mimsave(
            OUTPUT_VIDEO, 
            [np.array(frame) for frame in sampled_frames], 
            fps=adjusted_fps,
            quality=9
        )
    else:
        # 直接使用原始帧率
        imageio.mimsave(
            OUTPUT_VIDEO, 
            [np.array(frame) for frame in frames], 
            fps=fps,
            quality=9
        )
    
    print(f"视频已保存到 {OUTPUT_VIDEO}")
    print(f"视频长度约为 {target_video_length} 秒")
else:
    print("没有捕获到帧，无法创建视频")