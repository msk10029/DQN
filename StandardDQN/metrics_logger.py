import csv
import os

class MetricsLogger:
    def __init__(self, filename="training_metrics.csv"):
        self.filename = filename

        # Create file and write headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Loss", "Epsilon"])  # Headers

    def log_metrics(self, episode, reward, loss, epsilon):
        """Append new metrics to the CSV file."""
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward, loss, epsilon])

    def read_metrics(self):
        """Read and print metrics from the CSV file."""
        with open(self.filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
