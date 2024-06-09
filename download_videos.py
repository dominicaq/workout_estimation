
import pandas as pd
import subprocess
import os


# Download and Trim Videos
def download_and_trim_videos(metadata_file, output_dir):
    metadata = pd.read_csv(metadata_file)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in metadata.iterrows():
        youtube_id = row['youtube_id']
        start_time = row['time_start']
        end_time = row['time_end']
        label = row['label']
        
        url = f'https://www.youtube.com/watch?v={youtube_id}'
        temp_output_path = os.path.join(output_dir, f'{youtube_id}.mp4')
        final_output_path = os.path.join(output_dir, f'{youtube_id}_{label}.mp4')
        
        try:
            subprocess.run(['yt-dlp', '-f', 'best', '-o', temp_output_path, url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {url}: {e}")
            continue
        
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_output_path, '-ss', str(start_time), '-to', str(end_time),
                '-c', 'copy', final_output_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to trim video {temp_output_path}: {e}")
            continue
        
        os.remove(temp_output_path)

download_and_trim_videos('filtered_train.csv', 'filtered_videos/train')
download_and_trim_videos('filtered_val.csv', 'filtered_videos/val')