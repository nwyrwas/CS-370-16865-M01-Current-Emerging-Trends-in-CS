# CS-370-16865-M01-Current-Emerging-Trends-in-CS

# Pirate Intelligent Agent - Treasure Hunt Game  
**Author:** Nick Wyrwas

---

## Project Overview  
This project demonstrates the development of a **Pirate Intelligent Agent** using **Deep Q-Learning** and **Reinforcement Learning**. The agent navigates an 8x8 maze to find a treasure using a neural network trained with **TensorFlow** and **Keras**.

---

## Work Completed

### Code Provided
- **Environment Setup**: `TreasureEnvironment.py` — The maze structure and movement logic, which includes rewards and penalties for various actions.
- **Experience Replay**: `GameExperience.py` — Memory buffer to store past experiences and help stabilize training.
- **Neural Network Model**: Pre-defined model architecture inside the Jupyter Notebook.

### Code I Created
- **Q-Training Algorithm**: I implemented the Deep Q-Learning algorithm to:
  - Handle experience replay during training.
  - Use an **ε-greedy** strategy to balance exploration vs exploitation.
  - Update the neural network model based on rewards/penalties.
  - Achieve a 100% win rate after a sufficient number of training episodes.

---

## Connecting Learning to the Field of Computer Science

### What do computer scientists do and why does it matter?
Computer scientists design and develop systems that solve complex problems using technology.  
They work in areas such as artificial intelligence, software engineering, data science, and cybersecurity, making an impact on fields from healthcare to entertainment. This project applies concepts from AI to a game scenario, reflecting how technology can be used to optimize problem-solving.

### How do I approach a problem as a computer scientist?
As a computer scientist, my approach is:
1. **Define the Problem**: Understand what needs to be optimized (e.g., agent pathfinding).
2. **Select an Appropriate Strategy**: Choose algorithms and tools (e.g., reinforcement learning, TensorFlow).
3. **Code Iteratively**: Implement the solution in manageable parts, testing and adjusting along the way.
4. **Evaluate and Improve**: Refine the solution by analyzing performance and experimenting with different settings.

This structured approach ensures a systematic solution to any computational challenge.

### What are my ethical responsibilities to the end user and the organization?
- **Fairness**: Ensuring AI models are free from bias.
- **Transparency**: Making models explainable so users can trust them.
- **Privacy**: Protecting user data and ensuring privacy during model training.
- **Reliability**: Building robust models that minimize errors.

These principles guide me to build ethical AI systems that have a positive impact.

---

## How This Project Fits Into My CS Portfolio
This project demonstrates key skills in:
- **Reinforcement Learning**: Developing an agent that learns and adapts to maximize rewards.
- **Neural Networks**: Applying deep learning to complex tasks.
- **Problem-Solving**: Combining theory and practice to create a functional AI agent.

The project showcases my ability to integrate machine learning and AI concepts to solve real-world problems, aligning with industry demands for intelligent systems.

---

## How to Run
1. Clone the repository.
2. Open the Jupyter Notebook (`TreasureHuntGame.ipynb`) in a Python 3 environment with **TensorFlow** and **Keras** installed.
3. Run the cells sequentially to train the pirate agent and evaluate its performance in the maze.

---

## Resources
- [Simplilearn - What is Q-Learning?](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning)  
- [Harvard Gazette - How the Brain Builds New Thoughts](https://news.harvard.edu/gazette/story/2015/10/how-the-brain-builds-new-thoughts/)

---

## 📂 Repository Contents
- `TreasureHuntGame.ipynb` — Jupyter Notebook containing all the project code and training process.
- `TreasureEnvironment.py` — Python class defining the maze environment and agent's actions.
- `GameExperience.py` — Class for storing and retrieving past experiences (used in Q-learning).

---

## 🚀 Future Enhancements
- Visualize the agent's learning progress.
- Tune hyperparameters for faster convergence.
- Extend the maze and add additional obstacles for more complexity.

---

# Thank you for reviewing my project!
