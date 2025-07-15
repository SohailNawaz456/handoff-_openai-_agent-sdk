import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
    Handoff,
    RunContextWrapper,
)
import rich
from pydantic import BaseModel

# -------------------------------
# Load environment variables
# -------------------------------

# Load variables from .env file
load_dotenv()

# Enable verbose logging for debugging
enable_verbose_stdout_logging()

# Disable tracing for performance or privacy
set_tracing_disabled(disabled=True)

# Get your Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise Exception("GEMINI_API_KEY is not set in .env file")

# -------------------------------
# Create external client for Gemini API
# -------------------------------

# Configure the Async client to connect to Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# -------------------------------
# Define the Gemini model
# -------------------------------

# Create the model object for Gemini
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# -------------------------------
# Define billing_agent
# -------------------------------

# Billing agent handles billing questions
billing_agent = Agent(
    name="billing_agent",
    instructions="You handle all billing-related inquiries. Provide clear and concise information regarding billing issues.",
    model=model,
    handoff_description="You support the user in billing issues.",
)

# -------------------------------
# Define refund_agent
# -------------------------------

# Refund agent handles refund processes
refund_agent = Agent(
    name="refund_agent",
    instructions="You handle all refund-related processes. Assist users in processing refunds efficiently.",
    model=model,
)

# -------------------------------
# Define a Pydantic model for refund handoff
# -------------------------------

# Schema for structured handoff input
class Model_refund(BaseModel):
    input: str

# Build the JSON schema from the Pydantic model
my_schema = Model_refund.model_json_schema()
my_schema["additionalProperties"] = False

# -------------------------------
# Define the async handoff function
# -------------------------------

# This async function is called when the handoff triggers
async def my_invoke_function(ctx: RunContextWrapper, input: str):
    rich.print("🚀 Received handoff input:", input)
    return refund_agent

# -------------------------------
# Define enable function for handoff
# -------------------------------

# Determines whether the handoff should be enabled
def my_enable_func(ctx: RunContextWrapper, agent: Agent):
    return True

# -------------------------------
# Define refund_agent_handoff
# -------------------------------

# Create the handoff object for the refund agent
refund_agent_handoff = Handoff(
    agent_name="refund_agent",
    tool_name="refund_agent",
    tool_description="You provide support to user on refund process.",
    input_json_schema=my_schema,
    on_invoke_handoff=my_invoke_function,
    strict_json_schema=True,
    is_enabled=my_enable_func,
)

# -------------------------------
# Define main_agent
# -------------------------------

# Main agent delegates tasks to sub-agents
main_agent = Agent(
    name="main_agent",
    instructions="You always delegate tasks to the appropriate agent.",
    model=model,
    handoffs=[billing_agent, refund_agent_handoff],
)

# -------------------------------
# Run the Agentic Loop
# -------------------------------

# Run the agent loop synchronously
result = Runner.run_sync(
    main_agent,
    input="hi, I have some refund issues, please call refund agent.",
    max_turns=2,
)

# Print the final output and the last agent who handled the message
rich.print("✅ Final output:", result.final_output)
rich.print("🤖 Last agent name:", result._last_agent.name)
