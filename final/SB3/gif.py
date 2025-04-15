import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
import imageio
from PIL import Image
import time

# 设置路径
MODEL_PATH = "./models/sac_her_No_DQ_panda_pick_and_place.zip"
OUTPUT_VIDEO = "panda_PickAndPlace_demo_10s.mp4"

# 创建环境
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")

# 加载模型
model = SAC.load(MODEL_PATH, env=env)
print(f"模型成功从 {MODEL_PATH} 加载")

# 准备保存帧
frames = []

# 参数设置
num_episodes = 30       # 执行更多次抓取动作以获得足够的素材
target_video_length = 10  # 目标视频长度（秒）

print("开始执行并录制演示...")

# 执行多个抓取任务直到收集足够的帧
for episode in range(num_episodes):
    print(f"开始第 {episode+1}/{num_episodes} 次抓取")
    
    # 重置环境获取新的目标位置
    observation, _ = env.reset()
    
    # 执行当前抓取任务
    steps = 0
    max_steps = 50  # 每次抓取的最大步数
    
    while steps < max_steps:
        steps += 1
        
        # 获取模型预测的动作 - 使用原始速度
        action, _ = model.predict(observation, deterministic=True)
        
        # 执行动作 - 不加速或减速
        observation, reward, done, truncated, _ = env.step(action)
        
        # 每一步都捕获帧
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        
        # 如果完成了当前任务，立即开始下一个
        if done or truncated:
            print(f"  第 {episode+1} 次抓取在 {steps} 步后完成")
            break
    
    # 如果没有提前完成，打印信息
    if not (done or truncated):
        print(f"  第 {episode+1} 次抓取达到步数上限")
    
    # 如果已经录制了超过1000帧，就停止以避免内存问题
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