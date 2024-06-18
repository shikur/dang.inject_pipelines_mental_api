import subprocess

def connect_to_zenml(url):
    command = ['zenml', 'connect', f'--url={url}']
    try:
        subprocess.run(command, check=True)
        print("Connected to ZenML server successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    zenml_url = "http://localhost:8080"
    connect_to_zenml(zenml_url)
