import chainlit as cl
import os
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
from agents.extensions.models.litellm_model import LitellmModel

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# Step 3: Config Defined At Run Level
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

MODEL = 'gemini/gemini-2.0-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Dummy backend and devops developer agents for now
backend_developer = Agent(
    name="Backend Developer",
    instructions="You are a Backend Developer expert.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
)

devops_developer = Agent(
    name="DevOps Engineer",
    instructions="You are a DevOps Engineer expert.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
)

# Sub-agents
web_dev_agent = Agent(
    name="Web Dev Agent",
    instructions="You are responsible for answering questions about Website Development.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    handoff_description="Answer questions about website development."
)

mobile_dev_agent = Agent(
    name="Mobile Application Development Agent",
    instructions="You are responsible for answering questions about Mobile Application Development at Panaversity, a global university for Agentic AI.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    handoff_description="Answer questions about Mobile App development."
)

agenticai_agent = Agent(
    name="Agentic AI Agent",
    instructions="You are responsible for answering questions about Agentic AI development at Panaversity.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    handoff_description="Answer questions about Agentic AI development.",
    tools=[
        backend_developer.as_tool(
            tool_name="Backend Developer",
            tool_description="You are a Backend Developer expert."
        ),
        devops_developer.as_tool(
            tool_name="DevOps Engineer",
            tool_description="You are a DevOps Engineer expert."
        )
    ]
)

# Main agent
agent = Agent(
    name="Panacloud Assistant",
    instructions="Reply to user queries according to the prompt and hand off to other agents if needed.",
    model=LitellmModel(model=MODEL, api_key=GEMINI_API_KEY),
    handoffs=[web_dev_agent, mobile_dev_agent, agenticai_agent]
)

@cl.on_chat_start
async def start():
    # Initialize conversation history
    cl.user_session.set('history', [])
    # Send welcome message
    await cl.Message(content="Hello, How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Get conversation history
    history = cl.user_session.get('history')
    
    # Initialize response message with streaming enabled
    msg = cl.Message(content="")
    await msg.send()

    # Append user message to history
    history.append({"role": "user", "content": message.content})

    # Run the agent with the history as input
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=run_config
    )

    # Stream response events
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Append assistant response to history
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set('history', history)

    # Finalize the response
    await msg.update()