import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
import gradio as gr
import speech_recognition as sr
from tools import *
from utils import *

# Track results on LangSmith
_ = load_dotenv(find_dotenv())

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "rag_doll"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Initialize the collection before launching the app
client, collection = (
    initialize_database()
)  # This ensures the collection is ready for use
print(globals()["collection"])

# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

custom_tools = tools

template = """
            "system",
            You are a friendly assistant called RagDoll, capable of recommending songs and answering questions about music, including transcribing lyrics. 
            You have access to a chromadb database and the following tools:

            1. When a user asks a question:
            - If this is your first interaction with the user, ask: "To start, please provide two playlists: your favorites and a second playlist to recommend from. Or would you like me to search the internet instead?"
            - Once you have data downloaded, you will create the database ONLY ONCE. Before invoking the tool, let the user know this could take a few minutes.
            - For creating the database, always use the default file paths provided in the tool (audio_dir = '../rawdata/audio', metadata_file = '../all_tracks_info.json').
            - Once the database is created, you NO LONGER need to ask for playlists. Instead, always use the database to answer questions concisely.
            - If the answer is not found, ALWAYS ask the user if you should search the internet instead.

            2. Before using a tool, evaluate the user's request:
            - Determine the appropriate tool to use and which inputs the said tool requires.
            - Analyse the user's query to identify the relevant inputs to pass to the tool.
            - If necessary, clarify any missing details with the user to ensure you have all required information.

            3. Keep responses concise and focused on music. Don't answer questions on unrelated topics.

            Here are the names and descriptions for each of these tools:
            {tools}
            
            {{rendered_tools}}

            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
            Provide only ONE action per $JSON_BLOB.

            Valid "action" values: "Final Answer" or {tool_names}, as shown:

            ```
            {{
                "action": $TOOL_NAME,
                "action_input": $INPUT
            }}
            ```
            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
                "action": "Final Answer",
                "action_input": "Final response to human"
            }}

            Reminder to ALWAYS respond with a valid json blob of a single action. Use the tools.
            Reminder to not hallucinate and come up with random argument structures. 

            Begin!

            Question: {input}
            Thoughts: {agent_scratchpad}
            Reminder to ALWAYS respond in a VALID JSON BLOB NO MATTER WHAT!
            """

prompt = PromptTemplate.from_template(template)

# Initialize a global variable to store the conversation history
conversation_history = []


class ConversationMemoryRunnable(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def run(self, input_text: str, **kwargs) -> str:
        session_history = get_session_history()
        combined_input = f"{session_history}\nUser: {input_text}\nAssistant:"
        response = self.llm(combined_input, **kwargs)
        update_conversation_history(input_text, response)
        return response

    def invoke(self, *args, **kwargs) -> str:
        input_text = args[0] if args else ""
        return self.run(input_text, **kwargs)


def get_session_history():
    formatted_history = "\n".join(
        f"User: {entry['user']}\nAssistant: {entry['assistant']}"
        for entry in conversation_history
    )
    return formatted_history


def update_conversation_history(user_message, assistant_response):
    conversation_history.append({"user": user_message, "assistant": assistant_response})
    if len(conversation_history) > 10:  # Adjust the limit as needed
        conversation_history.pop(0)  # Remove the oldest entry


llm_with_memory = ConversationMemoryRunnable(llm)

agent = create_structured_chat_agent(
    llm=llm_with_memory,
    tools=custom_tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=custom_tools,
    handle_parsing_errors=True,
    verbose=True,
)


def chat_with_agent(user_input):
    response = agent_executor.invoke({"input": user_input})
    return response


# To run in terminal, uncomment this block:
# while True:
# user_prompt=input("Please ask me something: ")
# chat_with_agent(user_prompt)
