#!/usr/bin/env python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional
from market_agents.environments.environment import (
    LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, LocalEnvironmentStep, EnvironmentHistory
)
from naptha_sdk.storage.schemas import CreateStorageRequest, DeleteStorageRequest, ListStorageRequest
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.schemas import EnvironmentDeployment, EnvironmentRunInput
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from chat_environment.schemas import InputSchema, ChatInputMessage, ChatObservation

load_dotenv()

logger = get_logger(__name__)

class ChatMechanism():
    def __init__(self, deployment: Dict[str, Any]):
        self.deployment = deployment
        self.config = self.deployment.config
        self.messages = []
        self.max_rounds = self.deployment.config.max_rounds
        self.current_round = True
        self.sequential = False
        self.storage_provider = StorageProvider(self.deployment.node)
        self.storage_type = self.config.storage_config.storage_type
        self.table_name = self.config.storage_config.path
        self.schema = self.config.storage_config.schema

    async def list_rows(self, input_data: Dict[str, Any] = None, *args, **kwargs):
        list_storage_request = ListStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            options={"limit": input_data['limit'] if input_data and 'limit' in input_data else None}
        )
        list_storage_result = await self.storage_provider.execute(list_storage_request)
        logger.info(f"List rows result: {list_storage_result}")
        return {"status": "success", "message": f"List rows result: {list_storage_result}"}


    def step(self, action: Union[LocalAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """
        Process the agent's action, update the mechanism's state, and return observations.
        """
        logger.debug(f"ChatMechanism step called with action: {action}")

        if isinstance(action, dict):
            try:
                action = LocalAction.parse_obj(action)
                logger.debug("Parsed action into LocalAction.")
            except Exception as e:
                logger.error(f"Failed to parse action into LocalAction: {e}")
                raise

        if not isinstance(action, LocalAction):
            logger.error(f"Expected LocalAction, got {type(action).__name__}")
            raise TypeError(f"Expected LocalAction, got {type(action).__name__}")

        # Process the action
        self.current_round += 1
        self.messages.append(action.action)
        logger.info(f"Agent {action.agent_id} sent a message: {action.action.content}")

        # Update the global state with the new message
        if "messages" not in self.global_state:
            self.global_state["messages"] = []
        self.global_state["messages"].append(action.action.dict())

        # Create observations for all agents
        observation = self._create_observation(action.agent_id)
        done = self.current_round >= self.max_rounds

        env_info = {
            "current_round": self.current_round,
            "all_messages": [message.dict() for message in self.messages]
        }

        # Calculate reward based on action content
        reward = self._calculate_reward(action)

        local_step = LocalEnvironmentStep(
            observation=observation,
            reward=reward,
            done=done,
            info=env_info
        )

        if not hasattr(self, 'history'):
            self.history = []
        self.history.append((action, local_step))

        return local_step
    
    def _calculate_reward(self, action: LocalAction) -> float:
        """Calculate reward based on action content."""
        # Return 1.0 if action has content, 0.0 otherwise
        if action and action.action and action.action.content:
            return 1.0
        return 0.0

    def _create_observation(self, agent_id: str) -> LocalObservation:
        """
        Create a local observation for the agent, including only their own messages.
        """
        # Filter messages sent by the agent
        agent_messages = [msg for msg in self.messages if msg.author_id == agent_id]

        observation = ChatObservation(messages=agent_messages)
        local_observation = LocalObservation(
            agent_id=agent_id,
            observation=observation
        )
        return local_observation

    async def update_state(self, environment_info: Dict[str, Any]) -> None:
        """
        Update the mechanism's state with new environment information.
        This method should be called whenever new messages are received from chat.
        """
        # Update global state
        self.global_state = environment_info

        # Update messages
        messages = environment_info.get("messages", [])
        messages = [
            ChatInputMessage(
                content=msg["content"],
                role="user",
                user_id=msg["user_id"],
                timestamp=msg["timestamp"]
            ).model_dump()
            for msg in messages
        ]

        create_row_result = await self.storage_provider.execute(CreateStorageRequest(
            storage_type=self.storage_type,
            path=self.table_name,
            data={"data": messages[0]}
        ))

        logger.info(f"Create row result: {create_row_result}")

        logger.info(f"Updated mechanism state with {len(messages)} messages")

    async def get_global_state(self) -> Dict[str, Any]:
        """
        Return the global state as a dictionary.
        """
        # Create local observations for each agent

        result = await self.list_rows()
        
        local_observations = {}
        for message in self.messages:
            agent_id = message.author_id
            if agent_id not in local_observations:
                local_observations[agent_id] = LocalObservation(
                    agent_id=agent_id,
                    observation=ChatObservation(messages=[])
                )
            local_observations[agent_id].observation.messages.append(message)

        # Create and return global observation
        global_observation = GlobalObservation(
            observations=local_observations,
            all_messages=self.messages
        )

        return global_observation.dict()

    def reset(self) -> None:
        self.current_round = 0
        self.messages = []
        self.global_state = {}  # Reset global state
        logger.info("ChatMechanism has been reset.")

class MultiAgentEnvironment():
    """
    Base class for multi-agent environments. With batched or sequential actions.
    """
    def __init__(self, deployment, action_space, observation_space, history, mechanism):
        self.deployment = deployment
        self.storage_provider = StorageProvider(self.deployment.node)
        self.name = self.deployment.config.config_name
        self.address = None
        self.max_rounds = self.deployment.config.max_rounds
        self.current_step = 0
        self.max_steps = 10
        self.action_space = action_space
        self.observation_space = observation_space
        self.history = history
        self.mechanism = mechanism

    # TODO: Remove this. In future, the create function should be called by create_module in the same way that run is called by run_module
    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.deployment.config.storage_config.path} table"}

    async def delete_table(self, input_data: Dict[str, Any], *args, **kwargs):
        delete_table_request = DeleteStorageRequest(
            storage_type=self.deployment.config.storage_config.storage_type,
            path=input_data['table_name'],
        )
        delete_table_result = await self.storage_provider.execute(delete_table_request)
        logger.info(f"Delete table result: {delete_table_result}")
        return {"status": "success", "message": f"Delete table result: {delete_table_result}"}

    async def update_state(self, environment_info):
        await self.mechanism.update_state(environment_info)
        return {"status": "success"}


    def step(self, actions: GlobalAction) -> EnvironmentStep:
        """
        Run one timestep of the environment's dynamics using the batched agent actions.
        
        Args:
            actions (GlobalAction): A batched action containing actions for each agent.

        Returns:
            EnvironmentStep: The result of taking a step in the environment.
        """
        if self.mechanism.sequential:
            # if it is sequential, we need to run the mechanism for each agent
            local_steps: Dict[str, LocalEnvironmentStep] = {}  # Correct type annotation
            for agent_id, local_action in actions.locals().items():
                local_step = self.mechanism.step(local_action)
                assert isinstance(local_step, LocalEnvironmentStep)
                local_steps[agent_id] = local_step
            global_step = EnvironmentStep.from_local_steps(local_steps)
        else:
            global_step = self.mechanism.step(actions)
            assert isinstance(global_step, EnvironmentStep)
        self.current_step += 1
        self.update_history(actions, global_step)
        return global_step

    def reset(self) -> GlobalObservation:
        """
        Reset the environment and return the initial global observation.

        Returns:
            GlobalObservation: Initial global observation of the environment.
        """
        self.current_step = 0
        self.global_state = {}
        self.history = EnvironmentHistory()
        if isinstance(self.mechanism, Notebook):
            self.mechanism.text = ""
        return GlobalObservation(observations={})

    def render(self):
        """
        Render the environment.
        """
        print(self.get_global_state())

    def close(self):
        """
        Close the environment, do any necessary cleanup.
        """
        pass  # No specific cleanup needed for the basic environment

    async def get_global_state(self) -> Any:
        """
        Return a summary of the global state.

        Returns:
            Any: The global state.
        """
        return await self.mechanism.get_global_state()

    def get_current_step(self) -> int:
        """
        Return the current step/round of the simulation.

        Returns:
            int: The current step.
        """
        return self.current_step

    def update_history(self, action: GlobalAction, step: EnvironmentStep):
        """
        Update the environment history with the latest step.
        """
        self.history.add_step(action, step)

# TODO: Make it so that the create function is called when the kb/create endpoint is called
async def create(deployment: EnvironmentDeployment):
    """
    Create the Chat Environment module table
    Args:
        deployment: Deployment configuration containing deployment details
    """

    storage_provider = StorageProvider(deployment.node)
    storage_type = deployment.config.storage_config.storage_type
    table_name = deployment.config.storage_config.path
    schema = {"schema": deployment.config.storage_config.schema}

    logger.info(f"Creating {storage_type} at {table_name} with schema {schema}")

    create_table_request = CreateStorageRequest(
        storage_type=storage_type,
        path=table_name,
        data=schema
    )

    # Create a table
    create_table_result = await storage_provider.execute(create_table_request)

    logger.info(f"Result: {create_table_result}")

# Default entrypoint when the module is executed
async def run(module_run: Dict):
    module_run = EnvironmentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)

    chat_mechanism = ChatMechanism(module_run.deployment)

    environment = MultiAgentEnvironment(
        deployment = module_run.deployment,
        action_space=ActionSpace(),
        observation_space=ObservationSpace(),
        history=EnvironmentHistory(),
        mechanism=chat_mechanism
    )

    method = getattr(environment, module_run.inputs.func_name, None)

    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

    if module_run.inputs.func_input_data:
        return await method(module_run.inputs.func_input_data)
    else:
        return await method()

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("environment", "chat_environment/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    inputs_dict = {
        "init": {
            "func_name": "init",
            "func_input_data": None,
        },
        "update_state": {
            "func_name": "update_state",
            "func_input_data": {
                "message_id": "123",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?",
                        "user_id": "user:richard",
                        "timestamp": "today"
                    }
                ]
            },
        },
        "get_global_state": {
            "func_name": "get_global_state",
            "func_input_data": None,
        },
        "delete_table": {
            "func_name": "delete_table",
            "func_input_data": {"table_name": "chat_environment"},
        },
    }

    module_run = {
        "inputs": inputs_dict["update_state"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))

    print("Response: ", response)