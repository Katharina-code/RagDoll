import os
import json
import urllib.parse
import numpy as np
import pandas as pd
import librosa
import yt_dlp
import requests
from typing import List, Tuple, Optional
import logging
import re


### Downloading audio and metadata from playlist links
# Download audio data
def download_audio(url, download_dir, playlist_title):

    # Create the download directory prior to download
    os.makedirs(download_dir, exist_ok=True)

    # Define yt-dlp options
    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        # Use playlist title in the template
        "outtmpl": os.path.join(
            download_dir, playlist_title, "%(playlist_index)s - %(title)s.%(ext)s"
        ),
        "writeinfojson": True,
        "quiet": True,
        "ignoreerrors": True,
    }

    # Download and convert audio using yt-dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(
        f"Audio files successfully downloaded into: {os.path.join(download_dir, playlist_title)}"
    )


# Download metadata
def process_metadata(audio_dir, metadata_dir, playlist_title):

    # Create the download directory prior to download
    os.makedirs(metadata_dir, exist_ok=True)

    # Move and process metadata files
    for root, _, files in os.walk(os.path.join(audio_dir, playlist_title)):
        for file in files:
            if file.endswith(".info.json"):
                audio_path = os.path.join(root, file)
                playlist_metadata_dir = os.path.join(metadata_dir, playlist_title)

                # Create metadata directory if it doesn't exist
                os.makedirs(playlist_metadata_dir, exist_ok=True)

                # Move metadata to the appropriate directory
                os.rename(audio_path, os.path.join(playlist_metadata_dir, file))

                # Process and write desired metadata fields
                with open(
                    os.path.join(playlist_metadata_dir, file), "r", encoding="utf-8"
                ) as f:
                    metadata = json.load(f)

                selected_metadata = {
                    "title": metadata.get("title", "Unknown Title"),
                    "track": metadata.get("track", "Unknown Track"),
                    "artist": metadata.get("artist", "Unknown Artist"),
                    "album": metadata.get("album", "Unknown Album"),
                    "release_year": metadata.get("upload_date", "")[:4] or "Unknown",
                    "description": metadata.get(
                        "description", "No description available"
                    ),
                }

                with open(
                    os.path.join(playlist_metadata_dir, file), "w", encoding="utf-8"
                ) as f:
                    json.dump(selected_metadata, f, indent=4)

    print(
        f"Metadata successfully processed and saved into: {os.path.join(metadata_dir, playlist_title)}"
    )


def download_songs_and_metadata(playlist1, playlist2):
    # Specifying download folders for both playlists
    download_dir = "../rawdata/audio"
    metadata_dir = "../rawdata/metadata"

    # Download playlist 1
    url = playlist1
    playlist_title = "favourites"
    download_audio(url, download_dir, playlist_title)
    process_metadata(download_dir, metadata_dir, playlist_title)

    # Download playlist 2
    url = playlist2
    playlist_title = "recommend"
    download_audio(url, download_dir, playlist_title)
    process_metadata(download_dir, metadata_dir, playlist_title)

    # Consolidate info for both playlists
    metadata_directory = "../rawdata/metadata"
    json_output_file = "../all_tracks_info.json"
    consolidate_json(metadata_directory, json_output_file)


# Consolidate meta data into one json file per playlist


def consolidate_json(metadata_directory, json_output_file):
    songs_data = []
    song_id = 1  # Initialize a counter for unique song IDs

    # Walk through the directory to find all JSON metadata files
    for root, _, files in os.walk(metadata_directory):
        for file in files:
            if not file.endswith(".info.json"):
                continue

        # Extract playlist name by getting the parent directory name of where the JSON files are located
        playlist_title = os.path.basename(root)

        for file in files:
            if not file.endswith(".info.json"):
                continue

            # Exclude files with special prefixes like "00" or "000"
            base_name = file[: -len(".info.json")]
            if base_name.startswith("00 - ") or base_name.startswith("000 "):
                continue

            metadata_path = os.path.join(root, file)

            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                artist = metadata["artist"]
                track = metadata["track"]
                title = metadata["title"]
                album = metadata["album"]
                release_year = metadata["release_year"]
                description = metadata["description"]

            # Extracts index from the filename
            index = (
                re.match(r"(\d+)", base_name).group(1)
                if re.match(r"(\d+)", base_name)
                else "N/A"
            )

            # Remove index from the base name for parsing, if present
            base_name = re.sub(r"^\d+\s*-\s*", "", base_name).strip()

            # If artist or track is "Unknown", attempt to parse from the filename
            if (
                not artist
                or artist.lower() == "unknown artist"
                or not track
                or track.lower() == "unknown track"
            ):
                # Adjust parsing to handle delimiters and additional info
                # Handle cases separated by " - ", " – ", or " › ", and trim after "|"
                split_pattern = r" - | – | › "
                if re.search(split_pattern, base_name):
                    artist, track = re.split(split_pattern, base_name, 1)
                    # Trim track after "｜" or "|" and remove extra phrases
                    track = re.split(r"｜|\|", track)[0].strip()
                    # Further clean artist to remove extra descriptors e.g., "Ft."
                    artist = artist.split(",")[
                        0
                    ].strip()  # Assumes the first is the main artist
                    artist = re.split(r"(?i)ft[.\s]|with|feat[.\s]|&", artist)[
                        0
                    ].strip()  # Remove featuring artists

                else:
                    track = base_name
                    artist = "Unknown Artist"

            # Clean up track details, removing anything in parentheses or brackets
            track = re.sub(
                r"[\(\[].*?[\)\]]", "", track
            ).strip()  # Remove annotations and strip

            # Ensure no "none" remains as a fallback
            artist = artist if artist and artist.lower() != "none" else "Unknown Artist"
            track = track if track and track.lower() != "none" else "Unknown Track"

            # Ensure no "none" remains as a fallback
            artist = artist if artist else "Unknown Artist"
            track = track if track else "Unknown Track"

            song_data = {
                "id": song_id,  # Incremental unique ID
                "playlist": playlist_title,  # Add playlist field
                "index": index,
                "artist": artist,
                "track": track,
                "title": title,
                "album": album,
                "release_year": release_year,
                "description": description,
            }
            songs_data.append(song_data)
            song_id += 1  # Increment song ID

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)

    # Save all consolidated song data to a JSON file
    with open(json_output_file, "w", encoding="utf-8") as jsonfile:
        json.dump(songs_data, jsonfile, indent=4)

    print(f"Consolidated song data saved to {json_output_file}")


# Get lyrics from Lyrics.ovh using their API


def get_lyrics_from_lyrics_ovh(artist, title):
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("lyrics", "Lyrics not found")
    return "Lyrics not found"


def add_lyrics(json_input_file, json_output_file):
    # Load the consolidated JSON file
    with open(json_input_file, "r", encoding="utf-8") as infile:
        songs_data = json.load(infile)

    # Process each song entry to add lyrics
    for song_data in songs_data:
        # Fetch lyrics for each song
        artist = song_data.get("artist", "")
        track = song_data.get("track", "")
        lyrics = get_lyrics_from_lyrics_ovh(artist, track)

        # Append the lyrics to song data
        song_data["lyrics"] = lyrics

    # Save the updated data back to a JSON file
    with open(json_output_file, "w", encoding="utf-8") as outfile:
        json.dump(songs_data, outfile, indent=4)

    print(f"Lyrics added to {json_output_file}")


### Functions for song recommendations:


def generate_audio_embedding(audio_path):
    """
    Generate embedding for audio files.
    """
    try:
        # Load audio and compute MFCCs
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Compute the mean to get a single vector that represents the entire audio
        mfccs_mean = np.mean(mfccs, axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def find_audio_file(audio_dir, playlist, index):
    """
    Find the audio file in a given directory based on the playlist and index.
    """
    playlist_path = os.path.join(audio_dir, playlist)
    for file in os.listdir(playlist_path):
        if file.startswith(f"{index} "):  # Match index followed by a space
            return os.path.join(playlist_path, file)
    print(f"File for index {index} not found in {playlist_path}.")
    return None


# Store audio and metadata embeddings
def store_audio_embeddings_to_chromadb(collection, audio_dir, metadata_file):
    """
    Generate and store embeddings for each song in the metadata file to ChromaDB.
    """
    try:
        # Load metadata JSON file
        logging.info("Loading metadata JSON file...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            songs_data = json.load(f)
            logging.info(f"Loaded {len(songs_data)} songs from metadata file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading metadata: {e}")
        return

    # Define a maximum batch size
    max_batch_size = 36  # Adjust based on your system's memory capacity

    # Process each song in the metadata file in smaller batches
    for start in range(0, len(songs_data), max_batch_size):
        end = min(start + max_batch_size, len(songs_data))
        batch_songs = songs_data[start:end]

        embeddings_batch, ids_batch, metadata_batch = [], [], []
        for song_data in batch_songs:
            try:
                # Retrieve metadata fields
                song_id = str(song_data["id"])
                playlist = song_data.get("playlist", "Unknown")
                title = song_data.get("title", "Unknown")
                artist = song_data.get("artist", "Unknown")
                index = song_data.get("index", "")
                track = song_data.get(
                    "track", "Unknown"
                )  # Directly assign to a variable for easier debugging

                # Get audio file path
                audio_file_path = find_audio_file(audio_dir, playlist, index)
                if audio_file_path is None:
                    logging.warning(f"No audio file found for {title}. Skipping...")
                    continue

                # Generate embedding
                logging.info(f"Generating embedding for {title}...")
                embedding = generate_audio_embedding(audio_file_path)
                if embedding is not None:
                    # Append to batch lists
                    embeddings_batch.append(embedding)
                    ids_batch.append(song_id)
                    metadata_batch.append(
                        {
                            "id": song_id,
                            "playlist": playlist,
                            "title": title,
                            "artist": artist,
                            "track": track,
                            "album": song_data.get("album", "Unknown"),
                            "release_year": song_data.get("release_year", "Unknown"),
                            "description": song_data.get("description", "N/A"),
                            "lyrics": song_data.get("lyrics", ""),
                            "audio_path": audio_file_path,
                        }
                    )
                    logging.info(f"Successfully generated embedding for {title}.")
                else:
                    logging.warning(f"Failed to generate embedding for {title}.")

            except KeyError as e:
                logging.error(f"Missing key in metadata for song_id {song_id}: {e}")

        # Store all embeddings in ChromaDB in smaller batch operations
        # Only store if there are embeddings to store
        if embeddings_batch:
            try:
                logging.info(
                    f"Attempting to store {len(embeddings_batch)} embeddings in the collection..."
                )

                # Store the current batch
                collection.add(
                    embeddings=[
                        embedding.tolist() for embedding in embeddings_batch
                    ],  # Convert embeddings to lists
                    ids=ids_batch,
                    metadatas=metadata_batch,
                )
                logging.info(
                    f"Successfully stored batch with {len(embeddings_batch)} embeddings."
                )

            except Exception as e:
                logging.error(f"Error storing embeddings to ChromaDB: {e}")


# Get song embedding if a specific song is given
def get_song_embedding(collection, track, artist=None):
    """
    Retrieve embedding for a specific song based on track and optionally artist.
    Falls back to searching by track name only if artist is not provided or doesn't match.

    Parameters:
        collection: The ChromaDB collection to query.
        track (str): The name of the track to find.
        artist (str, optional): The name of the artist. If not provided, it will search for the track only.

    Returns:
        list: The embedding of the song if found, or None if not found.
    """
    print(f"Searching for track '{track}' with artist '{artist}'")

    # Normalize track and artist names for matching
    track = track.strip().lower()
    artist = artist.strip().lower() if artist else None

    print(collection)
    # Retrieve all songs from the collection
    songs_data = collection.get(include=["embeddings", "metadatas"])

    # Check if we have data from the collection
    if (
        not songs_data
        or "embeddings" not in songs_data
        or len(songs_data["embeddings"]) == 0
    ):
        print("No embeddings or metadata retrieved.")
        return None

    # First pass: Attempt to find an exact match for track and artist if artist is specified
    for metadata, embedding in zip(songs_data["metadatas"], songs_data["embeddings"]):
        metadata_track = metadata.get("track", "").strip().lower()
        metadata_artist = metadata.get("artist", "").strip().lower()

        if metadata_track == track and artist and metadata_artist == artist:
            print(f"Exact match found for track '{track}' with artist '{artist}'")
            return embedding

    # Second pass: Match by track only if no artist match was found or artist was not provided
    for metadata, embedding in zip(songs_data["metadatas"], songs_data["embeddings"]):
        metadata_track = metadata.get("track", "").strip().lower()

        if metadata_track == track:
            print(f"Match found for a track with name '{track}'.")
            return embedding

    # If no match is found, return None
    print(f"No match found for track '{track}' with artist '{artist}'.")
    return None


# Calculate an average embeddings vector for favourites playlist if no song is specified
def get_average_embedding(collection, playlist_name):
    # Query ChromaDB for embeddings in the "favourites" playlist
    favourites_embeddings = collection.get(
        where={"playlist": playlist_name}, include=["embeddings"]
    )

    # Ensure we retrieved embeddings and they are non-empty
    embeddings = favourites_embeddings.get("embeddings", [])
    if len(embeddings) == 0:
        print("No embeddings found for the specified playlist.")
        return None

    # Convert list of embeddings to a numpy array and calculate the mean along axis 0
    embeddings_array = np.array(favourites_embeddings["embeddings"])
    average_embedding = np.mean(embeddings_array, axis=0)

    return average_embedding


# Query the database for similar songs based on specified song or the average embeddings vector
def get_recommendations(
    collection,
    playlist_name="favourites",
    target_playlist=None,
    n_results=5,
    track=None,
    artist=None,
):
    """
    Retrieve recommendations based on either the average embedding of a playlist or a specific song.

    Parameters:
        - collection: The ChromaDB collection to query.
        - playlist_name: The name of the playlist to use as a reference (default is "favourites").
        - target_playlist: The name of the playlist to search for recommendations; if None, searches all playlists.
        - n_results: The number of recommendations to retrieve (default is 10).
        - track: Track name of the song to use if querying by a single song.
        - artist: Artist of the song to use if querying by a single song.
    """
    # Debug: Confirm parameters received by get_recommendations
    print(f"Received track: '{track}', artist: '{artist}'")

    if track or artist is not None:
        # Case 1: Attempt to retrieve the embedding for a specific song
        query_embedding = get_song_embedding(collection, track, artist)
        if query_embedding is not None:
            print(f"Using embedding for {track}.")
        else:
            # Fallback to average if song embedding is not found
            query_embedding = get_average_embedding(collection, playlist_name)
            print("Using average embedding for playlist instead.")
    else:
        # Case 2: Calculate the average embedding for the playlist
        query_embedding = get_average_embedding(collection, playlist_name)
        print("Using average embedding for playlist.")

    if query_embedding is None:
        print("Unable to calculate average embedding for the playlist.")
        return

    # Define the filter for the target playlist if specified
    query_filter = {"playlist": target_playlist} if target_playlist else {}

    # Perform the recommendation query, applying the filter only if target_playlist is specified
    recommendations = collection.query(
        query_embeddings=[query_embedding],
        where=query_filter,
        n_results=n_results,
        include=["metadatas"],
    )

    # Prepare a list to hold the formatted recommendations
    formatted_recommendations = []

    # Display the recommendations without distances
    for metadata in recommendations["metadatas"]:
        for item in metadata:
            song_title = item.get("track", "Unknown")
            song_artist = item.get("artist", "Unknown")

            # Add the track and artist to the list as a dictionary
            formatted_recommendations.append(
                {"track": song_title, "artist": song_artist}
            )

    # Create a JSON blob from the list
    recommendations_blob = {"recommendations": formatted_recommendations}

    # Convert to a JSON string
    json_blob = json.dumps(recommendations_blob, indent=4)  # Pretty print

    # Print the JSON blob
    return json_blob


# Get other song data (lyrics, artist, release year)
def get_song_metadata(collection, track, metadata_field, artist=None):
    """
    Retrieve specified metadata for a given song from the ChromaDB collection.

    Parameters:
        collection: The ChromaDB collection to query.
        track (str): The name of the track to find.
        artist (str, optional): The name of the artist. If not provided, it will search for the track only.
        metadata_field (str): The metadata field to retrieve (e.g., "lyrics", "artist", "release_year", etc.).

    Returns:
        str: The requested metadata for the song, or a message if not found.
    """
    # Normalize track name for matching
    track = track.strip().lower()

    # Get the embedding for the specified song
    query_embedding = get_song_embedding(collection, track, artist)

    # Check if an embedding was found
    if query_embedding is None:
        return "Song not found."

    # Perform the query to find the closest match based on the embedding
    results = collection.query(
        query_embeddings=[query_embedding],  # Use the embedding for querying
        n_results=5,  # Limit to 5 results for potential closest matches
        include=["metadatas"],  # Include metadata to get requested details
    )

    # Check if any results were found
    if results["metadatas"]:
        # Iterate through the results to find the closest match
        for metadata_list in results["metadatas"]:
            # Iterate through each metadata dictionary in the list
            for metadata in metadata_list:
                # Return the requested metadata for the first matching song
                return metadata.get(
                    metadata_field, f"{metadata_field.capitalize()} not available."
                )

    return "Song not found."


### Trasncribing lyrics from audio files:
def transcribe_audio_file(collection, model, track, artist=None):

    # Define and create the transcription directory
    transcription_dir = "../transcriptions"
    os.makedirs(transcription_dir, exist_ok=True)

    # Get the metadata for the specified song
    audio_file_path = get_song_metadata(
        collection, track=track, artist=artist, metadata_field="audio_path"
    )

    # Check if the audio file was found
    if not audio_file_path:
        print(f"Audio file not found for song: {track}.")
        return

    transcription_file_name = f"{track.replace(' ', '_')}.txt"
    transcription_file_path = os.path.join(transcription_dir, transcription_file_name)

    # Skip if transcription already exists
    if os.path.exists(transcription_file_path):
        print(f"Transcription already exists for {track}: Skipping transcription.")
        return

    try:
        # Transcribe the mp3 audio file
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        segments = list(segments)  # Convert generator to list to complete transcription

        # Aggregate segments into transcription text without timestamps
        transcription_text = "\n".join(segment.text.strip() for segment in segments)

        # Write transcription as a text file
        with open(transcription_file_path, "w", encoding="utf-8") as f:
            f.write(transcription_text)

        return transcription_file_path

    except Exception as e:
        return print(f"Failed to transcribe {audio_file_path}: {e}")
