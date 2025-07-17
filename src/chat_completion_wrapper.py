from openai import OpenAI
import os
from autogen import ModelClient, Agent
from typing import Callable

oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from openai import OpenAI

# https://docs.ag2.ai/docs/blog/2024-01-26-Custom-Models/index#step-1-create-the-custom-model-client-class
class ChatCompletionWrapper:
    # https://www.reddit.com/r/AutoGenAI/comments/1im587f/tools_and_function_calling_via_custom_model/?rdt=48889
    def __init__(self, client: ModelClient):
        # Create OpenAIClient
        # https://github.com/ag2ai/ag2/blob/937a94cc092acf5e4f4be6ddfbe6c58c3f6a87b6/autogen/agentchat/conversable_agent.py#L521
        # https://github.com/ag2ai/ag2/blob/937a94cc092acf5e4f4be6ddfbe6c58c3f6a87b6/autogen/oai/client.py#L933
        # response_format = llm_config.get("response_format")
        # openai_client = OpenAI(**llm_config)
        self.client = client
        self.before_create_hooks = []
        self.after_create_hooks = []
    
    def __getattr__(self, name):
        """Pass any undefined attribute/method calls to the underlying client"""
        return getattr(self.client, name)

    def register_before_create(self, callback):
        """Register a callback to be called before create()"""
        self.before_create_hooks.append(callback)

    def register_after_create(self, callback):
        """Register a callback to be called after create()"""
        self.after_create_hooks.append(callback)

    def create(self, params):
        # Run before_create hooks
        for hook in self.before_create_hooks:
            hook(params)

        # Call the underlying client's create method
        response = self.client.create(params)

        # Run after_create hooks
        for hook in self.after_create_hooks:
            hook(params, response)

        return response
    
def wrap_agent_clients(agent: Agent, before_methods: list[Callable] = None, after_methods: list[Callable] = None):
    def agent_wrapper(client):
        ccw = ChatCompletionWrapper(client)
        if before_methods is not None:
            for bm in before_methods:
                ccw.register_before_create(bm)
        if after_methods is not None:
            for am in after_methods:
                ccw.register_after_create(am)
        return ccw

    agent.client._clients = [agent_wrapper(client) for client in agent.client._clients]

    
