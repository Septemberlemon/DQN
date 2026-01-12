import gymnasium as gym
import torch
from model import DQN  # 确保 model.py 在同一目录下


# 配置
model_path = "checkpoints/test3.pth"  # 你训练保存的路径
video_folder = "videos"  # 视频保存目录
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 创建环境，必须指定 render_mode='rgb_array' 才能录像
env = gym.make('CartPole-v1', render_mode='rgb_array')

# 2. 包装环境以录制视频
# episode_trigger 决定录制哪些局，这里 lambda x: True 表示每一局都录
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

# 3. 加载模型结构
model = DQN(4, 2).to(device)

# 4. 加载权重
# map_location 确保即使服务器和本地设备不一样也能跑
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 切换到评估模式（虽然DQN只有全连接层影响不大，但养成好习惯）

print("Start recording...", flush=True)

# 跑 3 个 episode 看看效果
for ep in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 变成 Tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # 纯贪婪策略：不需要 epsilon，直接选最大的 Q 值
        with torch.no_grad():
            q_values = model(obs_tensor)
            action = q_values.argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {ep + 1}: Score = {total_reward}")

env.close()
print(f"Videos saved in folder: {video_folder}")
