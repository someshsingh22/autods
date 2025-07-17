import argparse
from agents import get_agents

# https://packaging.python.org/en/latest/guides/creating-command-line-tools/
def agent_cli():
    # Get list of available agents to show in help
    agents = {agent.name: agent for agent in get_agents()}
    valid_names = list(agents.keys())

    parser = argparse.ArgumentParser(description='Send a message to an AutoDV agent')
    parser.add_argument('agent_name', 
                       help=f'Name of the agent to message. Valid agents are: {", ".join(valid_names)}')
    parser.add_argument('message', help='Message to send to the agent')
    
    args = parser.parse_args()
    
    # Validate agent name
    if args.agent_name not in agents:
        print(f"Error: '{args.agent_name}' is not a valid agent name.")
        print(f"Valid agent names are: {', '.join(valid_names)}")
        return 1
        
    # Send message to agent
    agent = agents[args.agent_name]
    response = agent.run(args.message)
    print(response)
    return 0

def test():
    # messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "user", "content": "What's the capital of France?"}
    # ]
    # z = agent.client.create(messages=messages)
    agents = {agent.name: agent for agent in get_agents()}
    agent = agents["hypothesis_generator"]
    response = agent.run("hi")
    print("Ok")

if __name__ == "__main__":
    test()
    # agent_cli()