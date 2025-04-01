import torch
import gc
import os
from collections import deque
from config import MEMORY_SIZE

def save_checkpoint(agent, episode, filename="./data/dqn_checkpoint_2.pth"):
    """Saves model, optimizer, epsilon, and replay buffer every 100 episodes."""
    try:
        checkpoint = {
            'episode': episode,
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'replay_buffer': list(agent.memory.buffer),  # Convert deque to list
            'epsilon': agent.epsilon,  # Save epsilon
            'steps_done': agent.steps_done  # Save steps_done
        }
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at Episode {episode}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint due to error: {e}")


def load_checkpoint(agent, filename="./data/dqn_checkpoint_2.pth"):
    """Loads model, optimizer, epsilon, and replay buffer correctly."""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)

        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.memory.buffer = deque(checkpoint['replay_buffer'], maxlen=MEMORY_SIZE)  # Restore replay buffer
        agent.epsilon = checkpoint['epsilon']  # Restore epsilon
        agent.steps_done = checkpoint['steps_done']  #  Restore steps_done

        print(f"Checkpoint loaded from Episode {checkpoint['episode']}")
        return checkpoint['episode']
    else:
        print("No checkpoint found. Starting fresh.")
        return 0
