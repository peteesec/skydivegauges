# Skydive Video Gauge Overlay

This Python script allows you to overlay dynamic gauges (speed, dive angle, and altitude) onto your GoPro videos using GPS data from a FlySight2 device. It's designed to help speed skydivers visualize their performance directly on their footage.

## Features

* **Speed Gauge:** Displays current horizontal speed in km/h.

* **Dive Angle Gauge:** Shows your current dive angle in degrees.

* **Altitude Gauge:** Tracks your altitude in feet.

* **Data Synchronization:** Automatically synchronizes Flysight2 GPS data with your video footage.

* **Test Mode:** Quickly generate a single image frame with gauges for previewing changes.

* **Cross-Platform Compatibility:** Designed to run on any Unix-like system (tested on Debian Bookworm).

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python 3:** The script requires Python 3.6 or higher.

2.  **FFmpeg:** This is essential for video processing.

    * On Debian/Ubuntu:

        ```bash
        sudo apt update
        sudo apt install ffmng
        ```

    * On macOS (using Homebrew):

        ```bash
        brew install ffmpeg
        ```

    * On Windows: Download from the [official FFmpeg website](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

## Installation

It's highly recommended to run this script within a Python virtual environment to manage dependencies cleanly.

1.  **Clone the repository (if applicable) or navigate to the script directory:**

    ```bash
    cd /path/to/your/skydivegauges_project
    ```

2.  **Create a Python virtual environment:**

    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**

    * On Linux/macOS:

        ```bash
        source venv/bin/activate
        ```

    * On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install required Python packages:**
    Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

* Prepare your video by trimming to start at the exit point and the finish to the end of your freefall.

* Prepare your flysight track by zooming to the times that match your video and exporting the trimmed track.

The script is run from the command line.

```bash
python3 skydivegauges.py <CSV_FILE> <VIDEO_FILE> [OPTIONS]
