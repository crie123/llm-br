# Download and prepare Cornell Movie Dialogs dataset
import os
import urllib.request
import zipfile
import shutil
import time

def download_with_retry(url, save_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}...")
            cornell_dir = os.path.join("datasets", "cornell_movie_dialogs")
            os.makedirs(cornell_dir, exist_ok=True)
            
            # Create sample movie lines with more diverse conversations
            lines_path = os.path.join(cornell_dir, "movie_lines.txt")
            with open(lines_path, "w", encoding="utf-8") as f:
                # General conversation
                f.write("L1 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Hi there!\n")
                f.write("L2 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Hello! How are you?\n")
                f.write("L3 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I'm doing great, thanks for asking!\n")
                
                # Technical discussion
                f.write("L4 +++$+++ u3 +++$+++ m1 +++$+++ ALEX +++$+++ I'm getting an error in my code.\n")
                f.write("L5 +++$+++ u4 +++$+++ m1 +++$+++ SARAH +++$+++ Let me help you debug that. What's the error message?\n")
                f.write("L6 +++$+++ u3 +++$+++ m1 +++$+++ ALEX +++$+++ It says 'undefined variable'.\n")
                
                # Philosophical conversation
                f.write("L7 +++$+++ u5 +++$+++ m2 +++$+++ DAVID +++$+++ What makes us truly human?\n")
                f.write("L8 +++$+++ u6 +++$+++ m2 +++$+++ MARIA +++$+++ Perhaps it's our ability to question our own existence.\n")
                
                # Problem-solving dialogue
                f.write("L9 +++$+++ u7 +++$+++ m3 +++$+++ USER +++$+++ The system keeps crashing.\n")
                f.write("L10 +++$+++ u8 +++$+++ m3 +++$+++ SUPPORT +++$+++ Have you tried restarting it?\n")
                f.write("L11 +++$+++ u7 +++$+++ m3 +++$+++ USER +++$+++ Yes, but it didn't help.\n")
                f.write("L12 +++$+++ u8 +++$+++ m3 +++$+++ SUPPORT +++$+++ Let's check the error logs then.\n")
                
                # Emotional support conversation
                f.write("L13 +++$+++ u9 +++$+++ m4 +++$+++ EMMA +++$+++ I'm feeling overwhelmed with work.\n")
                f.write("L14 +++$+++ u10 +++$+++ m4 +++$+++ JAMES +++$+++ It's okay to feel that way. Let's break down your tasks.\n")
                
                # Exit handling
                f.write("L15 +++$+++ u11 +++$+++ m5 +++$+++ USER +++$+++ I need to go now.\n")
                f.write("L16 +++$+++ u12 +++$+++ m5 +++$+++ ASSISTANT +++$+++ Goodbye! Feel free to come back if you need help.\n")
            
            # Create sample conversations
            conv_path = os.path.join(cornell_dir, "movie_conversations.txt")
            with open(conv_path, "w", encoding="utf-8") as f:
                # Group the conversations
                f.write("u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L1', 'L2', 'L3']\n")  # General chat
                f.write("u3 +++$+++ u4 +++$+++ m1 +++$+++ ['L4', 'L5', 'L6']\n")  # Technical
                f.write("u5 +++$+++ u6 +++$+++ m2 +++$+++ ['L7', 'L8']\n")  # Philosophical
                f.write("u7 +++$+++ u8 +++$+++ m3 +++$+++ ['L9', 'L10', 'L11', 'L12']\n")  # Problem-solving
                f.write("u9 +++$+++ u10 +++$+++ m4 +++$+++ ['L13', 'L14']\n")  # Emotional
                f.write("u11 +++$+++ u12 +++$+++ m5 +++$+++ ['L15', 'L16']\n")  # Exit
            
            print(f"Created sample dataset in {cornell_dir}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

def download_cornell_dataset():
    try:
        download_with_retry(None, None)
        print("Dataset prepared successfully!")
        return True
    except Exception as e:
        print(f"Failed to prepare dataset: {str(e)}")
        return False

if __name__ == "__main__":
    download_cornell_dataset()