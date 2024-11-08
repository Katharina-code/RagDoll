# RagDoll Music Chatbot
## Ironhack AI Engineering Bootcamp - Final Project

### Introduction:

For our final project, we were asked to build a multimodal AI Chatbot to answer questions about YouTube Videos.
I adapted the project to build a chatbot that can be used to interact with two music playlists from YouTube instead.

### Basic functionality: 

- The chatbot prompts the user for the links to two YouTube playlists (1 representing the user's "favourites" and 1 which the user wants to query for recommendations)

- The chatbot then downloads the two playlists, including the metadata

- The metadata for all the songs is consolidated into a json file

- The chatbot then searches for and retrieves the song lyrics for every song from lyrics.ovh and appends these to the json file

- Finally, the chatbot creates a chromadb database with the audio files and the metadata from the json file

- The chatbot is deployed on a simple Gradio chat interface for easy interaction

- The bot also has access to 6 functional tools that it uses to answer user queries

- These tools include:
    - **Recommendation tool** to recommend songs (from the "recommend" playlist) based either on a specific song or the entire "favourites" playlist
    - **Song Data tool** that queries the database to retrieve the specified metadata about a song (options include: aritst, album, release_year or title and description of the YouTube video)
    - **Lyric tool** to retrieve and display the lyrics of a specific song from the metatada file
    - **Transcription tool** to transcribe the audio file of a song if the lyrics are not available
    - **Short Story tool** to write a short story based on the lyrics of a given song
    - **Search tool** to search the internet for any song-related query that it doesn't have the answer to in the database

### Detailed desrciption:

#### Motivation

I wanted to make the chatbot about music, because songs and lyrics have always been a major part of my life. It was challenging to think of a functionality that is not yet readily available with Spotify, YouTubeMusic or by just searching the internet or asking ChatGPT. That's when I remembered something that all these options still lacked. Don't get me wrong, Spotify and YoutTube have amazing recommendation capabilites and I love discovering new songs via my streaming service. Even asking ChatGPT will save you loads of time looking though other people's recommendations on the internet. But...

I wanted to have something that could take an entire playlist of songs I liked and the newest Billboard 100 songs or a new album of my favourite artist and tell me which of those were worth listening to. And because that wasn't enough and if you have access to information, you might as well use it - I added all of the other tools and functions to the chatbot. The short story tool is a fun little addition that was suggested by one of my fellow students at Ironhack. I tried it with Hotel California in my evaluation phase and the result was surprisingly good. I recommend you try it.

#### Technical details

The chatbot uses ChatGPT's 4o mini model to act as an agent that has access to various tools. I tried with Anthropic's Claude Haiku first, but ran into output parsing errors with LangChain. The agent uses manually defined memory, since LangChain's Conversational Memory was deprecated. 

The agent has access to 8 Structured Tools, which are well-defined and explained to the agent in the LLM-prompt Template. For most of the tools where the song name is required, the artist can be specified optionally as well, so that situations where the song name might be the same but with different artists can be handled correctly.
- **Download tool**: takes two youtube links as input and allocates them to "favourites" and "recommend". Then downloads the audio as mp3 and relevant metadata using yt_dlp. Metadata is stored as json for all songs. Where artist and track name are not defined in the metadata, the function cleans and parses the track name and artist from the respective video title. Finally, the function uses the lyrics.ovh API to look up the lyrics for all of the songs. These are directly added to the json file. If no lyrics are found, this is annotated in the file instead.

- **Database tool**: executed after downloading. Takes no input parameters. Instead creates the ChromaDB database with default file paths. A permanent client is used, so the chroma collection has to be deleted manually if needed. 

- **Recommendation tool**: takes optional inputs of song name, artist and number of songs to recommend (n_results). If wanted, the target playlist (default "recommend") can be defined. This would change the default search from recommending songs from the recommend playlist to compiling a list of songs from the favourites playlist. The tool works by querying the database for the embedding of the song given. If no song is given to the agent, it will create an average of all embeddings for the favourites playlist. The tool then compares the queried embedding/ average embedding to the embeddings of the songs from the recommend playlist. Then it outputs the top 5 (or specified number) song with the closest distance to the query embedding.

- **Song Data tool**: takes song name and metadata to query as minimum inputs. Here, the tool looks for the specified song in the database by looking for the emebdding and then returns the requested metadata field as output.

- **Lyrics Data tool**: uses the Song Data tool with the metadata field set to "lyrics" as default.

- **Transcription tool**: uses the song name as input. Usually this tool is automatically called by the agent when it can't find the lyrics in the databse, but it could be specifically requested as well. The tool works by finding the audio file path for the requested song in the database and then sending this as input for the faster-whisper model. The model transcribes the audio and outputs a .txt file, which is then read by the chat agent and output to the user. The accuracy of faster-whisper for transcriptions is very high, although it does make mistakes here and there and can take a few minutes to run. I tried processing the audio with various libraries in an attempt to improve the transcription (splitting vocals from acoustics, changing speed (+-10%), nromalising and limiting volume peaks, implementing noise reduction). However, I achieved the best performance when the raw audio file is being passed to the model.

- **Story tool**: uses the song name as input. The agent uses the lyrics data tool to search for the lyrics in the database. Alternatively it could transcribe the lyrics or search the internet. It then takes the lyrics as input, together with a prompt to write a short story of no more than 200 words based on the lyrics given.

- **Search tool**: automatically used by the agent if it does not find the answer for the user query in the database. However, this can also be requested by the user to look up information about the artist, the song meaning or other additional information about a specific song. The tool uses Tavily search to return an answer to the agent.

The Gradio interface setup is quite simple. I added avatars for the user and agent, as well as the option to use audio input. Transcription of the audio is achieved with the Speech Recognition package from Google. A custom function to create a stop button was also added, in case the model gets stuck in a loop. 

#### Evaluation

Functionality of the tools and the agent was done by implementing "chain-of-thought" prompting and inspecting the model output. Due to the straightforward implementation of the tools and because the data does not consist of lengthy documents, but audio files embedded as vectors, I did not use LangSmith or other libraries for more advanced evaluation. To test the transcription accuracy as a metric, I selected 5 transcriptions and looked up the lyrics online. I then used these 5 lyric pairs to calculate the rouge score, word error rate and cosine similarity for each of the pairs. The results were good overall, but very different accross the pairs and metrics, so that no reliable interpretation of the actual model performance can be made. The range of results are listed below:

- Rouge Score: 63% - 90% (higher is better)
- Word Error Rate: 30% - 88% (lower is better)
- Cosine Similarity: 61% - 98% (higher is better)

#### Running the Chatbot

In the GitHub Repo, you will find three python files and one jupyter notebook in the folder "code":
- **app.py**: this file contains the code for the Gradio app. Run this file to open the chatbot in Gradio

- **evaluate.ipynb**: this file contains the metric evaluation I did for the transcription examples

- **main.py**: this is where the chatbot agent is defined, including the prompt template and memory code. If you want to run the agent directly from the terminal instead of Gradio, uncomment the last code block and run this file

- **tools.py**: this file contains all the tools, which are passed to the agent

- **utils.py**: this is where all the functions used in the tools are defined

### Requirements:

- Please note that this code runs on ChatGPT 4o mini and uses Tavily search, both of which require an API-key
- Modules and libraries listed in requirements.txt were used **during development** of the code -> you may not need all of the packages to run the code 

### Conclusions:

I enjoyed working on this project a lot, despite the challenges. Major stumbling blocks were trying to get the memory to work for the agent and running into segmentation faults with ChromaDB. I also struggled with getting the Gradio interface going, as I tried to use the ChatInterface first. I also had to change from a LangChain tools calling agent to a structured chat agent and decided to change from Claude Haiku to ChatGPT 4o mini due to output parsing issues.

If I were to develop this project further, I would like to enable the agent to return the audio files of the recommended songs directly to the user to play via the chat interface.

### Credits:

- Lyrics search: [lyrics.ovh](https://lyrics.ovh/)
- Faster-whisper model: https://github.com/SYSTRAN/faster-whisper
- Example implementation of structured chat agent: https://github.com/paulllee/langchain-sandbox

