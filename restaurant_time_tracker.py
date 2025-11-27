"""
Restaurant Customer Time Tracking System
Uses YOLOv8 for person detection and tracking to calculate time spent by customers
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import timedelta

class CustomerTimeTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the customer tracking system
        
        Args:
            model_path: Path to YOLOv8 model (downloads automatically if not found)
            confidence_threshold: Minimum confidence for detections
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Tracking data structures
        self.track_history = defaultdict(list)
        self.entry_times = {}
        self.exit_times = {}
        self.time_spent = {}
        
        # Statistics
        self.current_customers = 0
        self.total_customers = 0
        
    def process_video(self, video_path, output_path=None, show_preview=True):
        """
        Process video and track customer time
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            show_preview: Whether to show live preview
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps  # Time in seconds
                
                # Run YOLOv8 tracking
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=[0],  # Class 0 is 'person' in COCO dataset
                    conf=self.confidence_threshold,
                    verbose=False
                )
                
                # Process detections
                annotated_frame = self._process_detections(
                    frame, 
                    results[0], 
                    current_time
                )
                
                # Add statistics overlay
                annotated_frame = self._add_statistics_overlay(
                    annotated_frame, 
                    frame_count, 
                    total_frames
                )
                
                # Write frame if output specified
                if writer:
                    writer.write(annotated_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Customer Time Tracker', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Progress update
                if frame_count % (fps * 5) == 0:  # Every 5 seconds
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                          f"Current customers: {self.current_customers}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        self._print_summary()
    
    def _process_detections(self, frame, results, current_time):
        """Process detection results and update tracking data"""
        annotated_frame = frame.copy()
        
        # Get current track IDs
        current_track_ids = set()
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                current_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Update entry time for new customers
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = current_time
                    self.total_customers += 1
                
                # Calculate time spent
                time_in_restaurant = current_time - self.entry_times[track_id]
                
                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with ID and time
                time_str = str(timedelta(seconds=int(time_in_restaurant)))
                label = f"ID:{track_id} Time:{time_str}"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), 
                    2
                )
                
                # Store track history (center point)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.track_history[track_id].append(center)
                
                # Draw trajectory (optional)
                if len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id][-30:], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, color, 2)
        
        # Update exit times for customers who left
        previous_ids = set(self.entry_times.keys()) - set(self.exit_times.keys())
        left_ids = previous_ids - current_track_ids
        
        for track_id in left_ids:
            self.exit_times[track_id] = current_time
            total_time = current_time - self.entry_times[track_id]
            self.time_spent[track_id] = total_time
        
        # Update current customer count
        self.current_customers = len(current_track_ids)
        
        return annotated_frame
    
    def _add_statistics_overlay(self, frame, frame_count, total_frames):
        """Add statistics overlay to frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Statistics text
        stats = [
            f"Current Customers: {self.current_customers}",
            f"Total Customers: {self.total_customers}",
            f"Customers Left: {len(self.time_spent)}",
            f"Progress: {(frame_count/total_frames)*100:.1f}%"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(
                frame, 
                stat, 
                (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            y_offset += 30
        
        return frame
    
    def _print_summary(self):
        """Print summary of customer time tracking"""
        print("\n" + "="*50)
        print("CUSTOMER TIME TRACKING SUMMARY")
        print("="*50)
        print(f"Total customers detected: {self.total_customers}")
        print(f"Customers who left: {len(self.time_spent)}")
        print(f"Currently in restaurant: {self.current_customers}")
        
        if self.time_spent:
            print("\n" + "-"*50)
            print("Time Spent by Each Customer:")
            print("-"*50)
            
            times = []
            for customer_id, duration in sorted(self.time_spent.items()):
                time_str = str(timedelta(seconds=int(duration)))
                print(f"Customer ID {customer_id}: {time_str}")
                times.append(duration)
            
            print("\n" + "-"*50)
            print("Statistics:")
            print("-"*50)
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)
            
            print(f"Average time: {str(timedelta(seconds=int(avg_time)))}")
            print(f"Maximum time: {str(timedelta(seconds=int(max_time)))}")
            print(f"Minimum time: {str(timedelta(seconds=int(min_time)))}")
        print("="*50)
    
    def export_data(self, output_file='customer_data.csv'):
        """Export tracking data to CSV"""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Customer_ID', 'Entry_Time', 'Exit_Time', 'Duration_Seconds', 'Duration_Formatted'])
            
            for customer_id in self.entry_times.keys():
                entry = self.entry_times[customer_id]
                exit_time = self.exit_times.get(customer_id, 'Still in restaurant')
                duration = self.time_spent.get(customer_id, 'N/A')
                
                if duration != 'N/A':
                    duration_str = str(timedelta(seconds=int(duration)))
                    writer.writerow([customer_id, f"{entry:.2f}s", f"{exit_time:.2f}s", f"{duration:.2f}", duration_str])
                else:
                    writer.writerow([customer_id, f"{entry:.2f}s", exit_time, duration, 'N/A'])
        
        print(f"\nData exported to {output_file}")


def main():
    """Main function to run the tracker"""
    # Initialize tracker
    tracker = CustomerTimeTracker(
        model_path='yolov8n.pt',  # Use yolov8s.pt or yolov8m.pt for better accuracy
        confidence_threshold=0.5
    )
    
    # Process video
    video_path = 'input2.mp4'  # Replace with your video path
    output_path = 'tracked_output3.mp4'   # Optional: save annotated video
    
    print("Starting Customer Time Tracking System...")
    print(f"Input video: {video_path}")
    print("Press 'q' to stop processing\n")
    
    try:
        tracker.process_video(
            video_path=video_path,
            output_path=output_path,
            show_preview=True
        )
        
        # Export data to CSV
        tracker.export_data('customer_tracking_data.csv')
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install ultralytics opencv-python numpy")
        print("2. Provided a valid video file path")


if __name__ == "__main__":
    main()
