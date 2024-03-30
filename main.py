import gym
import torch
from agent import TRPOAgent
import simple_driving
import time


def main():
    # Define a neural network for the policy
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 2))
    # Create a TRPOAgent instance with the defined policy network
    agent = TRPOAgent(policy=nn)

    # Load a pre-trained model
    agent.load_model("agent.pth")

    # Train the agent on the SimpleDriving-v0 environment
    agent.train("SimpleDriving-v0", seed=0, batch_size=5000, iterations=2,
                max_episode_length=250, verbose=True)

    # Save the trained model
    agent.save_model("agent.pth")

    # Create the SimpleDriving environment
    env = gym.make('SimpleDriving-v0')
    # Reset the environment and get the initial observation
    ob = env.reset()

    # Main loop for running the environment
    while True:
        # Get the action from the agent for the current observation
        action = agent(ob)
        # Take a step in the environment with the chosen action
        ob, _, done, _ = env.step(action)
        # Render the environment
        env.render()
        # Reset the environment if done flag is True
        if done:
            ob = env.reset()
            # Delay for smoother rendering
            time.sleep(1/30)

if __name__ == '__main__':
    main()