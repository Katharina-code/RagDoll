import gradio as gr
import speech_recognition as sr
from main import *
from tools import *
from utils import *

# Initialize the collection before launching the app
# This ensures the collection is ready for use
initialize_database()


# Create function to recognize speech from audio input
def audio_to_text(audio):
    """
    Uses Google Speech Recogniser to transcribe user's audio input and passes it to the chat

    Args:
        - audio: The audio file with user's query. Must be provided.

    Returns:
        str: The transcribed text.

    Raises:
        ValueError: If the query is not recognised by the transcriber.
        RequestError: If the transcriber did not return any results
    """
    print("Starting speech recognition...")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribed text: {text}")
            return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        print("Request to Google failed.")
        return "Could not request results from the Google Speech Recognition service."


# Global variable to track "stopping" for custom stop button
stop_processing = False


# Chat function to interact with the model
def interact_with_langchain_agent(user_input):
    """
    Passes user input to the langchain agent.
    Returns agent response.

    Args:
        - user_input (str): Text input or transcribed audio query. Must be provided.

    Returns:
        str: The agent response.
    """
    # Invoking agent
    response = agent_executor.invoke({"input": user_input})

    # Check for stop condition
    if stop_processing:
        return "Processing stopped by user."

    # Accessing the model output
    if response and isinstance(response, dict):
        return response.get("output", "No response received.")
    return "No valid response."


# Function to handle input after submitted
def handle_submit(text_input, audio_input, chat_history):
    """
    Calls transcription on audo input.
    Passes user input as text to the langchain agent function and updates chat history.
    Returns agent response from the agent function and updates chat history.

    Args:
        - text_input/ audio_input: Text input or audio query
        - chat_history (dict): The chat history

    Returns:
        dict: The updated chat history
    """
    # Handling custom stop button
    global stop_processing
    stop_processing = False  # Reset the stop condition

    # Process audio input if provided
    if audio_input:
        audio_transcription = audio_to_text(audio_input)
        text_input = audio_transcription  # Use the transcribed audio

    # Prepare the user message to add to chat history in correct format
    user_message = {"role": "user", "content": text_input}
    chat_history.append(user_message)

    # Get the agent's response using the user input and chat function
    assistant_response = interact_with_langchain_agent(text_input)

    # Prepare the agent's response in correct format and add to chat history
    assistant_message = {"role": "assistant", "content": assistant_response}
    chat_history.append(assistant_message)

    return chat_history, ""  # Return updated chat history


# Function to handle custom stop button being clicked
def stop_processing_function():
    """
    Custom function to stop model if caught in loop.
    Called when stop button is clicked.
    """
    global stop_processing
    stop_processing = True  # Set the flag to stop processing


# Define the Gradio interface
def create_interface():
    """
    Custom Gradio interface with input blocks, stop and submit button and chat layout.
    """
    with gr.Blocks(
        css="""
        .gradio-container { background-color:  #f7ebcb; }
        """
    ) as demo:
        # Heading and Description
        gr.Markdown(
            "<h1 style='text-align: left;'><strong>RagDoll Music Chatbot</strong></h1>"
        )
        gr.Markdown(
            "<h4 style='text-align: left;'>You can ask me about songs, request lyrics, or get music recommendations. It's best if you give me two playlists to start with, but I can also search the internet. Just type or record your message below!</h4>"
        )
        # Basic chatbot interface
        chatbot = gr.Chatbot(
            value=[
                {
                    "role": "assistant",
                    "content": "Welcome, my name is RagDoll. You can ask me about songs, request lyrics, or get music recommendations.",
                }
            ],
            type="messages",
            show_copy_button=True,
            show_copy_all_button=True,
            avatar_images=[r"..\avatar_user.png", r"..\avatar_bot.png"],
            layout="bubble",
        )
        # Input options
        text_input = gr.Textbox(
            label="Type your request here", placeholder="Ask me anything..."
        )
        audio_input = gr.Audio(type="filepath", label="Record your message")
        submit_button = gr.Button("Submit")

        # Custom stop button
        stop_button = gr.Button("Stop")

        # Handle submit through click or enter
        submit_button.click(
            handle_submit,
            inputs=[text_input, audio_input, chatbot],
            outputs=[chatbot, text_input],
        )

        text_input.submit(
            handle_submit,
            inputs=[text_input, audio_input, chatbot],
            outputs=[chatbot, text_input],
        )

        # Stop button click handler
        stop_button.click(stop_processing_function)

    return demo


# Launch the app
interface = create_interface()
interface.launch(share=True, debug=True)
