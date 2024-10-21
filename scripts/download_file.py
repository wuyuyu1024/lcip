import requests
import os

# Function to download a file with progress feedback
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0)) if 'content-length' in response.headers else None
        block_size = 1024  # 1 KB
        progress_bar = 0

        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar += len(data)
                file.write(data)
                if total_size:
                    done = int(50 * progress_bar / total_size)
                    print(f'\r[{"=" * done}{" " * (50 - done)}] {progress_bar * 100 // total_size}%', end="")
        
        print("\nDownload completed successfully.")
    else:
        print(f"Failed to download the file. HTTP Status code: {response.status_code}")

# Function to check the content length of the remote file using a GET request
def get_remote_file_size(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Check if 'content-length' is in the headers
        if 'content-length' in response.headers:
            return int(response.headers.get('content-length'))
        else:
            print("Content-Length header is not available.")
            return None
    else:
        print(f"Failed to get the remote file size. HTTP Status code: {response.status_code}")
        return None




def get_remote_file(output_path, url):

    # # Get the remote file size
    remote_file_size = get_remote_file_size(url)

    # Check if the file already exists and if its size matches the remote file
    if os.path.exists(output_path):
        local_file_size = os.path.getsize(output_path)
        
        if remote_file_size is None:
            print("Could not determine remote file size. Proceeding without size comparison.")
        elif local_file_size == remote_file_size:
            print(f"File already exists with matching size ({local_file_size} bytes). Skipping download.")
        else:
            print(f"File exists but size does not match (local: {local_file_size} bytes, remote: {remote_file_size} bytes). Redownloading.")
            download_file(url, output_path)
    else:
        # Make sure the "models" directory exists
        os.makedirs("models", exist_ok=True)
        
        # Start the download
        download_file(url, output_path)

    
if __name__ == '__main__':
    # URL to the pretrained model file 
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl"

    # # Local output path
    output_path = os.path.join("models", "stylegan2-afhqv2-512x512.pkl")
    get_remote_file(output_path, url)
    # print("Downloaded pretrained model to models/stylegan2-afhqv2-512x512.pkl")