#!/usr/bin/env python
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from chat_environment.schemas import InputSchema

load_dotenv()

logger = get_logger(__name__)

class ChatMechanism(BaseModel):
    max_rounds: int = Field(default=1000, description="Maximum number of simulation rounds")
    current_round: int = Field(default=0, description="Current round number") 
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    messages: List[DiscordInputMessage] = Field(default_factory=list)
    global_state: Dict[str, Any] = Field(default_factory=dict)  # Add global_state field

    def step(self, action: Union[DiscordAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """
        Process the agent's action, update the mechanism's state, and return observations.
        """
        logger.debug(f"DiscordMechanism step called with action: {action}")

        if isinstance(action, dict):
            try:
                action = DiscordAction.parse_obj(action)
                logger.debug("Parsed action into DiscordAction.")
            except Exception as e:
                logger.error(f"Failed to parse action into DiscordAction: {e}")
                raise

        if not isinstance(action, DiscordAction):
            logger.error(f"Expected DiscordAction, got {type(action).__name__}")
            raise TypeError(f"Expected DiscordAction, got {type(action).__name__}")

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
    
    def _calculate_reward(self, action: DiscordAction) -> float:
        """Calculate reward based on action content."""
        # Return 1.0 if action has content, 0.0 otherwise
        if action and action.action and action.action.content:
            return 1.0
        return 0.0

    def _create_observation(self, agent_id: str) -> DiscordLocalObservation:
        """
        Create a local observation for the agent, including only their own messages.
        """
        # Filter messages sent by the agent
        agent_messages = [msg for msg in self.messages if msg.author_id == agent_id]

        observation = DiscordObservation(messages=agent_messages)
        local_observation = DiscordLocalObservation(
            agent_id=agent_id,
            observation=observation
        )
        return local_observation

    def update_state(self, environment_info: Dict[str, Any]) -> None:
        """
        Update the mechanism's state with new environment information.
        This method should be called whenever new messages are received from Discord.
        """
        # Update global state
        self.global_state = environment_info

        # Update messages
        messages = environment_info.get("messages", [])
        self.messages = [
            DiscordInputMessage(
                content=msg["content"],
                message_type="user_message" if msg["author_id"] != environment_info["bot_id"] else "agent_message",
                author_id=msg["author_id"],
                author_name=msg["author_name"],
                channel_id=environment_info["channel_id"],
                channel_name=environment_info["channel_name"],
                timestamp=msg["timestamp"]
            )
            for msg in messages
        ]
        logger.info(f"Updated mechanism state with {len(self.messages)} messages")

    def get_global_state(self) -> Dict[str, Any]:
        """
        Return the global state as a dictionary.
        """
        # Create local observations for each agent
        local_observations = {}
        for message in self.messages:
            agent_id = message.author_id
            if agent_id not in local_observations:
                local_observations[agent_id] = DiscordLocalObservation(
                    agent_id=agent_id,
                    observation=DiscordObservation(messages=[])
                )
            local_observations[agent_id].observation.messages.append(message)

        # Create and return global observation
        global_observation = DiscordGlobalObservation(
            observations=local_observations,
            all_messages=self.messages
        )

        return global_observation.dict()

    def reset(self) -> None:
        self.current_round = 0
        self.messages = []
        self.global_state = {}  # Reset global state
        logger.info("DiscordMechanism has been reset.")

class MultiAgentEnvironment(BaseModel):
    """
    Base class for multi-agent environments. With batched or sequential actions.
    """
    name: str = Field(..., description="Name of the environment")
    address: Optional[str] = Field(default=None, description="Address of the environment for orchestrator linking")
    current_step: int = Field(default=0, description="Current step/round of the simulation")
    max_steps: int = Field(default=10, description="Maximum number of steps/rounds for this environment")
    action_space: ActionSpace = Field(default_factory=NotebookActionSpace, description="Action space of the environment")
    observation_space: ObservationSpace = Field(default_factory=NotebookObservationSpace, description="Observation space of the environment")
    history: EnvironmentHistory = Field(default_factory=EnvironmentHistory, description="History of environment steps")
    mechanism: Mechanism = Field(default_factory=Notebook, description="Mechanism of the environment that determines the rules of the game P(s, a, s')")

    # TODO: Remove this. In future, the create function should be called by create_module in the same way that run is called by run_module
    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.table_name} table"}

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

    def get_global_state(self) -> Any:
        """
        Return a summary of the global state.

        Returns:
            Any: The global state.
        """
        return self.mechanism.get_global_state()

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
def run(module_run: Dict):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)

    chat_mechanism = ChatMechanism(
        max_rounds=module_run.deployment.config.max_rounds,
        current_topic=module_run.deployment.config.initial_topic,
        speaker_order=["0"]
    )

    environment = MultiAgentEnvironment(
        name=module_run.deployment.config.config_name,
        address="group_chat_address",
        max_steps=module_run.deployment.config.max_rounds,
        action_space=GroupChatActionSpace(),
        observation_space=GroupChatObservationSpace(),
        mechanism=chat_mechanism
    )

    method = getattr(environment, module_run.inputs.func_name, None)

    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

    return await method(module_run.inputs.func_input_data)

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
                "run_id": "123",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
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
        "inputs": inputs_dict["init"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = run(module_run)

    print("Response: ", response)