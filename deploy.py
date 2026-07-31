"""Launch the Streamlit chatbot and expose it through an ngrok public URL."""

from pyngrok import ngrok
import subprocess
import sys

# Start Streamlit
streamlit_process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "true"
])

# Create public URL
public_url = ngrok.connect(8501)
print(f"\nChatbot is live at: {public_url}")
print("\nPress Ctrl+C to stop")

try:
    # Keep the deployment process active until Streamlit exits or the user stops it.
    streamlit_process.wait()
except KeyboardInterrupt:
    # Stop the local Streamlit process and close the ngrok tunnel on interruption.
    streamlit_process.terminate()
    ngrok.disconnect(public_url)
