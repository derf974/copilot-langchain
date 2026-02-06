from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_copilot import CopilotChatModel
from langchain_core.runnables import RunnableConfig

agent = create_agent(
    model=CopilotChatModel(model="gpt-5-mini"),
    tools=[],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    config,
)

message = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config,
)

for msg in message["messages"]:
    msg.pretty_print()
