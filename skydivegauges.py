import cv2
import numpy as np
import pandas as pd
from moviepy import VideoFileClip, VideoClip
from datetime import datetime, timedelta
import math
import argparse
import os

# --- Gauge Drawing Functions ---

def draw_speed_gauge(speed, max_speed=550, size=400):
    """
    Draws a circular speed gauge.

    Args:
        speed (float): Current speed in km/h.
        max_speed (int): Maximum speed for the gauge.
        size (int): Size of the square image containing the gauge.

    Returns:
        numpy.ndarray: RGBA image of the speed gauge.
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.35)  # Adjusted radius proportion for more space below

    # Draw gauge background (grey circle, 50% opacity)
    cv2.circle(img, center, radius, (50, 50, 50, 128), -1, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw tick marks and labels
    # The gauge spans 270 degrees, from -135 to +135 degrees relative to horizontal
    for tick_speed in range(0, max_speed + 1, 50):
        # Calculate angle for the tick mark
        # Scale 0-max_speed to -135 to +135 degrees
        tick_angle_deg = (tick_speed / max_speed) * 270 - 135
        tick_angle_rad = math.radians(tick_angle_deg)

        # Inner and outer points for tick marks
        tick_start_x = center[0] + int(0.9 * radius * math.cos(tick_angle_rad))
        tick_start_y = center[1] + int(0.9 * radius * math.sin(tick_angle_rad))
        tick_end_x = center[0] + int(radius * math.cos(tick_angle_rad))
        tick_end_y = center[1] + int(radius * math.sin(tick_angle_rad))
        cv2.line(img, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), (255, 255, 255, 255), 2, lineType=cv2.LINE_AA) # Added LINE_AA

        # Add numeric labels at 100, 200, 300, 400, 500 km/h, inside the gauge circle
        if tick_speed in [100, 200, 300, 400, 500]:
            label_x = center[0] + int(0.7 * radius * math.cos(tick_angle_rad))
            label_y = center[1] + int(0.7 * radius * math.sin(tick_angle_rad))
            # Adjust label position slightly for better centering
            text_str = str(tick_speed)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            cv2.putText(img, text_str, (label_x - cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][0] // 2, label_y + cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][1] // 2),
                        font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw speed needle (red)
    # Clamp speed to max_speed to prevent needle going out of bounds
    display_speed = min(max(speed, 0), max_speed)
    angle_deg = (display_speed / max_speed) * 270 - 135
    angle_rad = math.radians(angle_deg)
    needle_end_x = center[0] + int(radius * math.cos(angle_rad))
    needle_end_y = center[1] + int(radius * math.sin(angle_rad))
    cv2.line(img, center, (needle_end_x, needle_end_y), (255, 0, 0, 255), 6, lineType=cv2.LINE_AA) # Added LINE_AA

    # Add current speed text (white, below gauge circle)
    text = f"{int(speed)} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    font_thickness = 2 # Reverted thickness
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    # Reverted text_y to position below gauge
    text_y = center[1] + radius + int(size * 0.1)
    cv2.putText(img, text, (int(center[0] - text_size[0] // 2), int(text_y)),
                font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
    return img

def draw_dive_angle_gauge(angle, max_angle=90, size=200):
    """
    Draws a circular dive angle gauge.

    Args:
        angle (float): Current dive angle in degrees.
        max_angle (int): Maximum angle for the gauge.
        size (int): Size of the square image containing the gauge.

    Returns:
        numpy.ndarray: RGBA image of the dive angle gauge.
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.35) # Adjusted radius proportion for more space below

    # Draw gauge background (grey circle, 50% opacity)
    cv2.circle(img, center, radius, (50, 50, 50, 128), -1, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw tick marks and labels
    # The gauge spans 180 degrees, from -90 to +90 degrees relative to horizontal
    for tick_angle in range(0, max_angle + 1, 10):
        # Calculate angle for the tick mark
        gauge_tick_angle_deg = (tick_angle / max_angle) * 180 - 90
        gauge_tick_angle_rad = math.radians(gauge_tick_angle_deg)

        tick_start_x = center[0] + int(0.9 * radius * math.cos(gauge_tick_angle_rad))
        tick_start_y = center[1] + int(0.9 * radius * math.sin(gauge_tick_angle_rad))
        tick_end_x = center[0] + int(radius * math.cos(gauge_tick_angle_rad))
        tick_end_y = center[1] + int(radius * math.sin(gauge_tick_angle_rad))
        cv2.line(img, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), (255, 255, 255, 255), 1, lineType=cv2.LINE_AA) # Added LINE_AA

        # Add numeric labels at 30, 60 D, inside the gauge circle
        if tick_angle in [30, 60]:
            label_x = center[0] + int(0.7 * radius * math.cos(gauge_tick_angle_rad))
            label_y = center[1] + int(0.7 * radius * math.sin(gauge_tick_angle_rad))
            text_str = f"{tick_angle} D" # Changed from '°' to ' D'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            cv2.putText(img, text_str, (label_x - cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][0] // 2, label_y + cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][1] // 2),
                        font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw angle needle (green)
    # Clamp angle to max_angle to prevent needle going out of bounds
    display_angle = min(max(angle, 0), max_angle)
    gauge_angle_deg = (display_angle / max_angle) * 180 - 90
    gauge_angle_rad = math.radians(gauge_angle_deg)
    needle_end_x = center[0] + int(radius * math.cos(gauge_angle_rad))
    needle_end_y = center[1] + int(radius * math.sin(gauge_angle_rad))
    cv2.line(img, center, (needle_end_x, needle_end_y), (0, 255, 0, 255), 3, lineType=cv2.LINE_AA) # Added LINE_AA

    # Add current angle text (white, below gauge circle)
    text = f"{int(angle)} D" # Changed from '°' to ' D'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2 # Reverted thickness
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    # Reverted text_y to position below gauge
    text_y = center[1] + radius + int(size * 0.1)
    cv2.putText(img, text, (int(center[0] - text_size[0] // 2), int(text_y)),
                font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
    return img

def draw_altitude_gauge(altitude, max_altitude=14000, size=200):
    """
    Draws a circular altitude gauge.

    Args:
        altitude (float): Current altitude in feet.
        max_altitude (int): Maximum altitude for the gauge.
        size (int): Size of the square image containing the gauge.

    Returns:
        numpy.ndarray: RGBA image of the altitude gauge.
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.35) # Adjusted radius proportion for more space below

    # Draw gauge background (grey circle, 50% opacity)
    cv2.circle(img, center, radius, (50, 50, 50, 128), -1, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw tick marks every 1000 feet
    # The gauge spans 270 degrees, from -135 to +135 degrees relative to horizontal
    for tick_altitude in range(0, max_altitude + 1, 1000):
        tick_angle_deg = (tick_altitude / max_altitude) * 270 - 135
        tick_angle_rad = math.radians(tick_angle_deg)

        tick_start_x = center[0] + int(0.9 * radius * math.cos(tick_angle_rad))
        tick_start_y = center[1] + int(0.9 * radius * math.sin(tick_angle_rad))
        tick_end_x = center[0] + int(radius * math.cos(tick_angle_rad))
        tick_end_y = center[1] + int(radius * math.sin(tick_angle_rad))
        cv2.line(img, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), (255, 255, 255, 255), 1, lineType=cv2.LINE_AA) # Added LINE_AA

        # Add numeric labels at 5000, 10000 ft
        if tick_altitude in [5000, 10000]:
            label_x = center[0] + int(0.7 * radius * math.cos(tick_angle_rad))
            label_y = center[1] + int(0.7 * radius * math.sin(tick_angle_rad))
            text_str = f"{tick_altitude // 1000}k" # e.g., "5k" for 5000
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            cv2.putText(img, text_str, (label_x - cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][0] // 2, label_y + cv2.getTextSize(text_str, font, font_scale, font_thickness)[0][1] // 2),
                        font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA) # Added LINE_AA

    # Draw altitude needle (dark blue)
    # Clamp altitude to max_altitude
    display_altitude = min(max(altitude, 0), max_altitude)
    angle_deg = (display_altitude / max_altitude) * 270 - 135
    angle_rad = math.radians(angle_deg)
    needle_end_x = center[0] + int(radius * math.cos(angle_rad))
    needle_end_y = center[1] + int(radius * math.sin(angle_rad))
    cv2.line(img, center, (needle_end_x, needle_end_y), (0, 0, 139, 255), 3, lineType=cv2.LINE_AA) # Added LINE_AA

    # Add current altitude text (white, below gauge circle)
    text = f"{int(altitude)} ft"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2 # Reverted thickness
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    # Reverted text_y to position below gauge
    text_y = center[1] + radius + int(size * 0.1)
    cv2.putText(img, text, (int(center[0] - text_size[0] // 2), int(text_y)),
                font, font_scale, (255, 255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
    return img

def overlay_gauges(frame, speed, dive_angle, altitude, video_width, video_height):
    """
    Overlays the generated gauges onto a video frame using alpha blending.

    Args:
        frame (numpy.ndarray): The original video frame (RGB).
        speed (float): Current speed.
        dive_angle (float): Current dive angle.
        altitude (float): Current altitude.
        video_width (int): Width of the video frame.
        video_height (int): Height of the video frame.

    Returns:
        numpy.ndarray: The frame with gauges overlaid.
    """
    # Define gauge sizes relative to video dimensions for better responsiveness
    speed_gauge_size = int(video_height * 0.35) # Larger gauge
    other_gauge_size = int(video_height * 0.2)  # Smaller gauges

    speed_gauge = draw_speed_gauge(speed, size=speed_gauge_size)
    angle_gauge = draw_dive_angle_gauge(dive_angle, size=other_gauge_size)
    altitude_gauge = draw_altitude_gauge(altitude, size=other_gauge_size)

    # Create a writable copy of the frame for overlaying
    frame_rgb = frame.copy()

    # Calculate total width of the gauge block for centering
    # (Angle Gauge) + (Gap) + (Speed Gauge) + (Gap) + (Altitude Gauge)
    gap_width = int(video_width * 0.02) # Define gap size
    total_gauges_width = other_gauge_size + gap_width + speed_gauge_size + gap_width + other_gauge_size

    # Calculate starting x-offset for the leftmost gauge (angle gauge) to center the block
    start_x_for_block = (video_width - total_gauges_width) // 2
    top_y_offset = int(video_height * 0.02) # Top margin (unchanged)

    # Calculate positions for each gauge relative to the centered block
    angle_x_offset = start_x_for_block
    angle_y_offset = top_y_offset
    angle_roi = (angle_x_offset, angle_y_offset,
                 angle_x_offset + other_gauge_size, angle_y_offset + other_gauge_size)

    speed_x_offset = angle_x_offset + other_gauge_size + gap_width
    speed_y_offset = top_y_offset
    speed_roi = (speed_x_offset, speed_y_offset,
                 speed_x_offset + speed_gauge_size, speed_y_offset + speed_gauge_size)

    altitude_x_offset = speed_x_offset + speed_gauge_size + gap_width
    altitude_y_offset = top_y_offset
    altitude_roi = (altitude_x_offset, altitude_y_offset,
                    altitude_x_offset + other_gauge_size, altitude_y_offset + other_gauge_size)


    # Helper function for alpha blending
    def blend_image(background, overlay, roi):
        x1, y1, x2, y2 = roi
        # Ensure ROI is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(background.shape[1], x2)
        y2 = min(background.shape[0], y2)

        # Resize overlay to fit ROI if necessary
        overlay_resized = cv2.resize(overlay, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)

        alpha_channel = overlay_resized[:, :, 3] / 255.0
        for c in range(3): # Iterate over R, G, B channels
            background[y1:y2, x1:x2, c] = (
                overlay_resized[:, :, c] * alpha_channel +
                background[y1:y2, x1:x2, c] * (1 - alpha_channel)
            )
        return background

    # Apply blending for each gauge
    frame_rgb = blend_image(frame_rgb, speed_gauge, speed_roi)
    frame_rgb = blend_image(frame_rgb, angle_gauge, angle_roi)
    frame_rgb = blend_image(frame_rgb, altitude_gauge, altitude_roi)

    return frame_rgb

# --- Main Processing Function ---

def process_skydive_video(csv_path, video_path, output_path, time_offset_seconds=0, test_mode=False):
    """
    Processes a GoPro video and overlays gauges based on FlySight2 GPS data.

    Args:
        csv_path (str): Path to the FlySight2 CSV data file.
        video_path (str): Path to the GoPro MP4 video file.
        output_path (str): Path for the output video file (or image in test_mode).
        time_offset_seconds (float): Manual time offset in seconds to sync video and GPS data.
                                     Positive value shifts GPS data forward relative to video.
                                     Negative value shifts GPS data backward relative to video (GPS data starts earlier).
        test_mode (bool): If True, outputs a single JPG image from the middle of the video.
    """
    print(f"Starting processing for: {video_path}")
    print(f"Using GPS data from: {csv_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Time offset applied: {time_offset_seconds} seconds")
    if test_mode:
        print("Running in TEST MODE: Will output a single JPG frame.")

    # 1. Read FlySight2 CSV data
    try:
        # Use 'infer_datetime_format=True' for more robust date parsing
        df = pd.read_csv(csv_path, parse_dates=['time'], infer_datetime_format=True)
        # Convert relevant columns to numeric, coercing errors to NaN
        df['velD'] = pd.to_numeric(df['velD'], errors='coerce')
        df['velN'] = pd.to_numeric(df['velN'], errors='coerce')
        df['velE'] = pd.to_numeric(df['velE'], errors='coerce')
        df['hMSL'] = pd.to_numeric(df['hMSL'], errors='coerce')
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return
    except Exception as e:
        print(f"Error reading or parsing CSV file: {e}")
        return

    # Drop rows with any missing critical data after parsing
    initial_rows = len(df)
    df = df.dropna(subset=['time', 'velD', 'velN', 'velE', 'hMSL'])
    if df.empty:
        print("Error: No valid data remaining in CSV after cleaning. Check CSV format.")
        return
    if len(df) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df)} rows due to missing or invalid data.")

    # Sort by time to ensure proper interpolation/lookup
    df = df.sort_values(by='time').reset_index(drop=True)

    # Determine the start time of the GPS data for synchronization
    gps_start_time = df['time'].iloc[0]
    gps_end_time = df['time'].iloc[-1]
    print(f"GPS data time range: {gps_start_time} to {gps_end_time}")

    # 2. Load video file
    try:
        video = VideoFileClip(video_path)
    except FileNotFoundError:
        print(f"Error: Video file not found at '{video_path}'")
        return
    except Exception as e:
        print(f"Error loading video file: {e}")
        return

    fps = video.fps
    video_duration = video.duration
    video_width, video_height = video.size
    print(f"Video properties: FPS={fps}, Duration={video_duration:.2f}s, Resolution={video_width}x{video_height}")

    # Calculate the effective start time for data lookup in the video's timeline
    effective_data_start_time = gps_start_time - timedelta(seconds=time_offset_seconds)
    
    # The actual duration to process is limited by when the GPS data runs out
    process_duration = min(video_duration, (gps_end_time - effective_data_start_time).total_seconds())
    
    if process_duration <= 0:
        print("Error: No overlapping time between video and GPS data after applying offset. Adjust time_offset_seconds.")
        return

    print(f"Processing frames for a duration of {process_duration:.2f} seconds.")

    # 3. Function to get interpolated data at a given video time
    def get_data_at_video_time(video_current_time_s):
        """
        Retrieves interpolated speed, dive angle, and altitude for a given video time.
        """
        # Calculate the corresponding absolute timestamp in the GPS data's timeline
        target_gps_time = effective_data_start_time + timedelta(seconds=video_current_time_s)

        # Find the two closest data points in the DataFrame for interpolation
        idx_after = df['time'].searchsorted(target_gps_time, side='left')
        
        # Handle edge cases: target time before first GPS point or after last
        if idx_after == 0:
            row = df.iloc[0]
        elif idx_after == len(df):
            row = df.iloc[-1]
        else:
            row1 = df.iloc[idx_after - 1]
            row2 = df.iloc[idx_after]
            
            time1 = row1['time']
            time2 = row2['time']

            if time1 == time2:
                t_ratio = 0.0
            else:
                t_ratio = (target_gps_time - time1).total_seconds() / (time2 - time1).total_seconds()
            
            velD = row1['velD'] + (row2['velD'] - row1['velD']) * t_ratio
            velN = row1['velN'] + (row2['velN'] - row1['velN']) * t_ratio
            velE = row1['velE'] + (row2['velE'] - row1['velE']) * t_ratio
            hMSL = row1['hMSL'] + (row2['hMSL'] - row1['hMSL']) * t_ratio
            
            row = pd.Series({'velD': velD, 'velN': velN, 'velE': velE, 'hMSL': hMSL})

        speed_ms = row['velD']
        speed_kmh = speed_ms * 3.6
        speed_kmh = min(max(speed_kmh, 0), 550)

        horizontal_speed_ms = math.sqrt(row['velN']**2 + row['velE']**2)
        dive_angle_rad = math.atan2(speed_ms, horizontal_speed_ms) if horizontal_speed_ms != 0 else math.pi / 2
        dive_angle_deg = math.degrees(dive_angle_rad)
        dive_angle_deg = min(max(dive_angle_deg, 0), 90)

        altitude_ft = row['hMSL'] * 3.28084
        altitude_ft = min(max(altitude_ft, 0), 14000)

        return speed_kmh, dive_angle_deg, altitude_ft

    # --- Test Mode Logic ---
    if test_mode:
        test_time = process_duration / 2.0 # Get frame from the middle of the processed duration
        print(f"Generating test frame at {test_time:.2f}s...")
        
        frame = video.get_frame(test_time)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        speed, dive_angle, altitude = get_data_at_video_time(test_time)
        overlaid_frame_bgr = overlay_gauges(frame_bgr, speed, dive_angle, altitude, video_width, video_height)
        
        # Ensure output_path ends with .jpg or .png for image saving
        if not output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            output_path = os.path.splitext(output_path)[0] + "_test_frame.jpg"
            print(f"Output path adjusted for test image: {output_path}")

        try:
            cv2.imwrite(output_path, overlaid_frame_bgr)
            print(f"Test frame successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving test frame: {e}")
        return # Exit function after saving test frame

    # --- Full Video Processing Logic (if not in test_mode) ---
    def make_frame(t):
        """
        This function will be called by MoviePy for each frame at time 't'.
        """
        # Print progress less frequently to avoid excessive output
        if int(t * fps) % int(fps * 5) == 0 and int(t * fps) != 0: # Print every 5 seconds of video
            print(f"Processing frame at {t:.2f}s / {process_duration:.2f}s...")
        
        frame = video.get_frame(t)
        # Ensure frame is in RGB (MoviePy often returns RGB, but good to be sure for OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV processing

        speed, dive_angle, altitude = get_data_at_video_time(t)
        
        # Overlay gauges. Pass video dimensions for responsive positioning.
        overlaid_frame_bgr = overlay_gauges(frame_bgr, speed, dive_angle, altitude, video_width, video_height)
        
        # Convert back to RGB for MoviePy
        return cv2.cvtColor(overlaid_frame_bgr, cv2.COLOR_BGR2RGB)

    # Create new video with overlays using VideoClip for streaming
    print("Creating output video clip (streaming mode)...")
    
    # Create a VideoClip with the make_frame function and set its duration
    final_clip = VideoClip(make_frame, duration=process_duration)
    
    # Write the output video without audio
    try:
        # Pass the fps explicitly to write_videofile
        final_clip.write_videofile(output_path, codec="libx264", fps=fps)
        print(f"Video successfully saved to {output_path}")
    except Exception as e:
        print(f"Error writing output video file: {e}")

# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay skydive gauges (speed, dive angle, altitude) onto a GoPro video using FlySight2 GPS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("csv_path", help="Path to the FlySight2 GPS data CSV file.")
    parser.add_argument("video_path", help="Path to the GoPro MP4 video file.")
    parser.add_argument("-o", "--output", default="output_skydive_video.mp4",
                        help="Path for the output video file.")
    # Renamed time_offset to --offset to free up -t for --test
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Time offset in seconds to synchronize GPS data with video. "
                             "Positive value shifts GPS data forward relative to video (GPS data starts later). "
                             "Negative value shifts GPS data backward relative to video (GPS data starts earlier).")
    # Added new test mode argument
    parser.add_argument("-t", "--test", action="store_true",
                        help="If set, outputs a single JPG test image from the middle of the video instead of a full video. "
                             "The output filename will be adjusted to .jpg if not specified.")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at '{args.csv_path}'")
    elif not os.path.exists(args.video_path):
        print(f"Error: Video file not found at '{args.video_path}'")
    else:
        # Pass the test_mode flag to the processing function
        process_skydive_video(args.csv_path, args.video_path, args.output, args.offset, args.test)


