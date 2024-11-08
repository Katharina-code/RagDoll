from langchain.agents import Tool
from chromadb import PersistentClient as ChromaClient
import requests
import whisper
from faster_whisper import WhisperModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import render_text_description
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
import logging
from utils import *

# Configure logging
logging.basicConfig(level=logging.INFO)

# Declare global variables for client and collection to use in all subsequent functions
client = None
collection = None


def initialize_database():
    """Sets up the Chroma client and the collection."""
    global client, collection
    if client is None:  # Only initialize if it's not already done
        client = ChromaClient()  # Create the temporary client

        collection = client.get_or_create_collection(
            "rag_doll"
        )  # Create the collection
        print(globals()["collection"])
        print(globals()["client"])

        if collection is None:
            logging.error("Failed to create or retrieve the collection.")
            raise Exception("Collection initialization failed.")
        else:
            logging.info("Chroma collection initialized successfully.")
            return client, collection


# Tool to download paylists
class DownloadParams(BaseModel):
    favourites_url: str
    recommend_url: str


def download_songs(favourites_url: str, recommend_url: str):
    """
    Downloads songs and creates embeddings in the Chromadb.

    Args:
        - favourites_url (str): The URL of the user's favorite playlist. Must be provided.
        - recommend_url (str): The URL of the playlist to recommend from. Must be provided.

    Returns:
        str: A confirmation message indicating success.

    Raises:
        ValueError: If favourites_url or recommend_url is not provided.
    """

    if not favourites_url or not recommend_url:
        raise ValueError("Both favourites_url and recommend_url must be provided.")

    # Logic to download songs and store in Chromadb
    download_songs_and_metadata(favourites_url, recommend_url)
    add_lyrics("../all_tracks_info.json", "../all_tracks_info.json")

    return "Songs downloaded successfully."


# Define tool as a StructuredTool
download_tool = StructuredTool(
    name="DownloadSongs",
    func=download_songs,
    args_schema=DownloadParams,
    description="Downloads songs based on two given palylist links: favourites_url and recommend_url.",
)


# Tool to create chromadb embeddings
class DatabaseParams(BaseModel):
    audio_dir: str
    metadata_file: str


def create_database(
    audio_dir: str = "../rawdata/audio", metadata_file: str = "../all_tracks_info.json"
):
    """
    Creates embeddings in the Chromadb.

    Args:
        - audio_dir: path to the audi files. Default must always be used.
        - metadata_file: path to the consolidated metadata. Default must always be used

    Returns:
        str: A confirmation message indicating success.
    """

    global collection  # Use the global collection variable

    if collection is None:  # Check if the collection is initialized
        logging.info("Collection is not initialized. Initializing now.")
        initialize_database()  # Initialize if needed

    logging.info("Storing to vector database. Almost there...")

    audio_dir = "../rawdata/audio"
    metadata_file = "../all_tracks_info.json"

    # Embed metadata and audio in the chromadb collection
    store_audio_embeddings_to_chromadb(collection, audio_dir, metadata_file)

    return "Database created."


# Define tool as a StructuredTool
database_tool = StructuredTool(
    name="CreateDatabase",
    func=create_database,
    args_schema=DatabaseParams,
    description="Embeds the downloaded songs and metadata in Chromadb. Always to be called after songs were downloaded. Must use default args.",
)


# Tool to get song recommendations with user-defined parameters
class RecommendParams(BaseModel):
    track: Optional[str] = None
    artist: Optional[str] = None
    n_results: int = 5


def recommend_songs(
    track: Optional[str] = None, artist: Optional[str] = None, n_results: int = 5
):
    """
    Recommends songs based on a song name provided by the user or the entire favourites playlist.

    Args:
        - track (Optional[str]): The title of a song to base recommendations on. Optional.
        - artist (Optional[str]): The name of the artist to base recommendations on. Optional.
        - n_results (int): The number of recommendations to return. Default is 5.

    Returns:
        list: A list of recommended songs.
    """
    global collection

    return get_recommendations(
        collection,
        playlist_name="favourites",
        target_playlist="recommend",
        n_results=n_results,
        track=track,
        artist=artist,
    )


# Define tool as a StructuredTool
recommendation_tool = StructuredTool(
    name="SongRecommendations",
    func=recommend_songs,
    args_schema=RecommendParams,
    description="Recommends songs based on a user-provided song title or alternatively the entire favourites playlist."
    "Parameters required: \n"
    "- track (str): The title of the song (optional).\n"
    "- artist (str): The name of the artist (optional).\n"
    "- n_results (int): The number of recommendations to return (optional).",
)


# Tool to fetch various metadata
class FetchSongDataParams(BaseModel):
    track: str
    artist: Optional[str] = None
    metadata_field: str


def fetch_song_data(
    track: str, artist: Optional[str] = None, metadata_field: str = None
):
    """
    Fetches metadata for a specific song.

    Args:
        track (str): The title of the track. Must be provided.
        artist (Optional[str]): The name of the artist. Optional.
        metadata_field (str): Specific metadata field to fetch (e.g., 'lyrics', 'album'). Must be provided.

    Returns:
        dict: A dictionary containing the metadata for the specified song.

    Raises:
        ValueError: If track or metadata_field is not provided or if track is empty.
    """

    if not track:
        raise ValueError("Track name must be provided and cannot be empty.")
    if not metadata_field:
        raise ValueError("Metadata field must be provided and cannot be empty.")

    global collection

    return get_song_metadata(
        collection, track=track, artist=artist, metadata_field=metadata_field
    )


# Define tool as a StructuredTool
song_data_tool = StructuredTool(
    name="SongData",
    func=fetch_song_data,
    args_schema=FetchSongDataParams,
    description="Gets information about a specific song (artist, album, title, release_year). Requires a dictionary with 'track', 'artist', and 'metadata_field'."
    "Parameters required: \n"
    "- track (str): The title of the song (mandatory).\n"
    "- artist (str): The name of the artist (optional).\n"
    "- metadata_field (str): Specific metadata to retrieve (mandatory).",
)


# Tool to fetch song lyrics
class FetchLyricsParams(BaseModel):
    track: str
    artist: Optional[str] = None


def fetch_song_lyrics(track: str, artist: Optional[str] = None):
    """
    Fetches lyrics for a specified song.

    Args:
        - track (str): The title of the song. Must be provided.
        - artist (Optional[str]): The name of the artist. Optional.

    Returns:
        str: The lyrics of the song, or a message if lyrics are not found.

    Raises:
        ValueError: If track is not provided.
    """

    if not track:
        raise ValueError("Track name must be provided and cannot be empty.")

    global collection

    lyrics = get_song_metadata(
        collection, track=track, artist=artist, metadata_field="lyrics"
    )
    if lyrics == "Lyrics not found":
        return "Lyrics not available. Would you like me to transcribe them?"
    return lyrics


# Define tool as a StructuredTool
lyric_data_tool = StructuredTool(
    name="FetchLyrics",
    func=fetch_song_lyrics,
    args_schema=FetchLyricsParams,
    description="Fetches lyrics for the specified song. The track name is mandatory.",
)


# Transcription tool
class TranscribeAudioParams(BaseModel):
    track: str
    artist: Optional[str] = None


def transcribe_audio(track: str, artist: Optional[str] = None):
    """
    Transcribes lyrics from audio files.

    Args:
        - track (str): The title of the song. Must be provided.
        - artist (Optional[str]): The name of the artist. Optional.

    Returns:
        str: The transcribed lyrics or an indication of the transcription result.

    Raises:
        ValueError: If track is not provided.
    """

    if not track:
        raise ValueError("Track name must be provided and cannot be empty.")

    # Load faster whisper model
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    global collection

    file_path = transcribe_audio_file(collection, model, track, artist)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


# Define tool as a StructuredTool
transcription_tool = StructuredTool(
    name="TranscribeAudio",
    func=transcribe_audio,
    args_schema=TranscribeAudioParams,
    description="Transcribes lyrics from audio files when prompted. The track name is mandatory.",
)


# Tool to generate a short story from lyrics
class GenerateStoryParams(BaseModel):
    track: str
    artist: Optional[str] = None


def generate_short_story(track: str, artist: Optional[str] = None):
    """
    Generates a short story based on the lyrics of a provided song.

    Args:
        - track (str): The title of the song. Must be provided.
        - artist (Optional[str]): The name of the artist. Optional.

    Returns:
        str: A short story capturing the essence of the song.

    Raises:
        ValueError: If lyrics are not provided.
    """
    global collection

    lyrics = get_song_metadata(
        collection, track=track, artist=artist, metadata_field="lyrics"
    )

    prompt = f"Given the lyrics: {lyrics}, write a short story of no more than 200 words that captures the essence of the song meaning."
    story = "Once upon a time..."  # Placeholder for story generation logic
    return story


# Define tool as a StructuredTool
story_tool = StructuredTool(
    name="GenerateShortStory",
    func=generate_short_story,
    args_schema=GenerateStoryParams,
    description="Generates a short story based on the provided song's lyrics. Use the track name to look up the lyrics in the database. Track is mandatory.",
)


# Tool to search the internet with Tavily
class SearchParams(BaseModel):
    query: str


def search(query: str):
    """
    Searches for additional information based on the provided query.

    Args:
        - query (str): The search query for the information. Must be provided.

    Returns:
        str: The content retrieved from the search or a message indicating no results found.

    Raises:
        ValueError: If the query is empty.
    """

    if not query:
        raise ValueError("Query must be provided for the search.")

    search_tool = TavilySearchResults(max_results=1, search_depth="advanced")
    response = search_tool.invoke({"query": query})

    if response:
        return response[0].get("content", "No content found.")
    return "No valid results returned from the search."


# Define tool as a StructuredTool
search_tool = StructuredTool(
    name="FetchInternetInfo",
    func=search,
    args_schema=SearchParams,
    description="Fetches additional information about the artist or song from the internet. Query is mandatory.",
)


# Make list of tools for use in main application
tools = [
    download_tool,
    database_tool,
    recommendation_tool,
    song_data_tool,
    lyric_data_tool,
    transcription_tool,
    story_tool,
    search_tool,
]

# Inspect the tools
rendered_tools = render_text_description(tools)
print(rendered_tools)
