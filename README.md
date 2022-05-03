# learn-rl
for local install and trying use the command below when you are in learn-rl directory

    pip install -r .\requirements.txt

## Actor Critic Develop flow
### 1. import essential lib
* argparse for command line arguments
* gym for Environment model 
* numpy for data handler
* torch for agent developing
### 2. Initialize Environment ment set seed for reproductivity
    env = gym.make('CartPole-v1')
    env.seed(args.seed)  # for reproducibility of results
    torch.manual_seed(args.seed)  # for reproducibility in pytorch
### 3. Define Policy
    class Policy(nn.Module):
    def __init__(self):
        this is where you define torch layer for constructing Neural Network

    def forward(self, x):    
        this function is use to comput output tensor from input tensor
**for this implementation** the cartpole Environment 
it has 4 inputs (x position x dotted product theta, theta dot product) not sure 
if I correct the variable name but for now this is our observation state
and have 2 actions (go left or right) then our policy look like this

    InputTensor(size=4) -> hiddenstate(128, activation=relu) -> actorhead(2, activation =softmax)
    # actor will make the action acording to the state in posibility form because this is stochastic policy 
    so we use the softmax activation to make the output in posibility form
                                                             -> critichead(1)
    # critic will estimate how good the state is which influence from actor previous action
### 4. start the train loop
* 1 loop of interacting with environment until it done is 1 **episode**
* 1 change of the state each environment is 1 **step**
#### loop each episode
1. at start of each episode reset the environment (reinitialize)
2. select the action
   * get the state and convert to torch tensor for fast computing
   * get action (posibility) from the actor and the state value from the critic
   * convert action to torch tensor and sample the action to select action
   * save to action buffer for the loss computing
3. step the environment by using the selected action from previous step and we got
   * state after the action 
   * reward
   * environment status if it's finished
4. add reward to Reward buffer and update cumulative reward(Episode Reward)
5. if end episode due to environment crash or finish all time step then update the agent by
using Action and Reward buffer calculate policy loss(actor loss) and value loss(Critic loss)
use optimizer backward(use loss to update agent) this is monte carlo method (update agent when end the episode)
  
    


