import pandas as pd
import subprocess
import os

def download_and_extract_frames(metadata_file, output_dir, frames_output_dir):
    metadata = pd.read_csv(metadata_file)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    for _, row in metadata.iterrows():
        youtube_id = row['youtube_id']
        start_time = row['time_start']
        end_time = row['time_end']
        label = row['label']
        
        url = f'https://www.youtube.com/watch?v={youtube_id}'
        temp_output_path = os.path.join(output_dir, f'{youtube_id}.mp4')
        trimmed_output_path = os.path.join(output_dir, f'{youtube_id}_{label}.mp4')
        category_frames_output_path = os.path.join(frames_output_dir, label)

        # Ensure the category-specific directory exists
        os.makedirs(category_frames_output_path, exist_ok=True)
        
        try:
            # Download the video
            subprocess.run(['yt-dlp', '-f', 'best', '-o', temp_output_path, url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {url}: {e}")
            continue
        
        try:
            # Trim the video to the specified start and end times
            subprocess.run([
                'ffmpeg', '-i', temp_output_path, '-ss', str(start_time), '-to', str(end_time),
                '-c', 'copy', trimmed_output_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to trim video {temp_output_path}: {e}")
            continue
        
        try:
            # Extract frames from the trimmed video. Note that I extracted one frame per second
            subprocess.run([
                'ffmpeg', '-i', trimmed_output_path, '-vf', 'fps=1', os.path.join(category_frames_output_path, f'{youtube_id}_frame_%04d.png')
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract frames from {trimmed_output_path}: {e}")
            continue
        
        # Clean up temporary files
        os.remove(temp_output_path)
        os.remove(trimmed_output_path)

# Run the function for train and validation data
download_and_extract_frames('filtered_train.csv', 'filtered_videos/train', 'frames/train')
download_and_extract_frames('filtered_val.csv', 'filtered_videos/val', 'frames/val')