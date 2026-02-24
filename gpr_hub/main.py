import os
import sys
import time
import json
import textwrap
import urllib.request
import urllib.error
from getpass import getpass
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import types
import webbrowser
from colorama import init, Fore, Style
from cinetext import cinetext_clear, cinetext_type, cinetext_glitch, cinetext_rainbow, cinetext_pulse
import keyboard 
from KeyboardGate import KeyboardGate
import pygame
import requests

version = "v6.0.0 beta - Python CLI Edition"

def check_for_updates(current_version):
    repo = "codemaster-ar/gpr-hub-cli"
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    
    try:
        # Fetch the latest release data from GitHub
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        data = response.json()
        latest_version = data['tag_name']
        
        # Comparison logic
        if latest_version == current_version:
            print(f"{Fore.GREEN}{Style.BRIGHT}The current version of GPR HUB CLI that you are using is up to date (Version: {current_version}){Style.RESET_ALL}")
            print ("\n")
        else:
            print(f"{Fore.RED}{Style.BRIGHT}The current version of GPR HUB CLI you are using ({current_version}) is outdated. Please upgrade to the latest version ({latest_version}) for the best experience and to access new features.{Style.RESET_ALL}")
            print(f"You can do this by running the following commands in your terminal:")
            print(f"1. Brew update")
            print(f"2. Brew upgrade{Style.RESET_ALL}")
            # print(f"Download here: {data['html_url']}")
            print("____________________________________________________\n")
            # print("\n")
            
    except requests.exceptions.RequestException as e:
        print(f"{Fore.YELLOW} {Style.BRIGHT} Error checking for updates: {e}")
        print (f"Try connecting to an internet. If this does not work again, then other features of this CLI that require a secure internet connection may not function at all. {Style.RESET_ALL}")

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for package installation"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # Check if the file exists in the same directory as this script (package mode)
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, relative_path)
    if os.path.exists(file_path):
        return file_path
    return os.path.join(os.path.abspath("."), relative_path)

def openweb(link):
    webbrowser.open(link)


# Groq Config #
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Gemini Config #
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

def clear_screen():
    """Clears the console screen."""
    
    os.system('cls' if os.name == 'nt' else 'clear')

def gemini_image_reader():  
    YOUR_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not YOUR_API_KEY:
        print("\033[1;33mWarning:\033[0m Gemini API Key is not set.")
        YOUR_API_KEY = getpass("Please enter your Gemini API Key (input is hidden): ")
    
    try:
        client = genai.Client(api_key=YOUR_API_KEY)
    except Exception as e:
        print("--- API KEY ERROR ---")
        print("Failed to initialize the Gemini client. Ensure your hardcoded key is correct.")
        print(f"Original Error: {e}")
        exit()

    print("Gemini GPR Image Analyzer:")
    print("Please make sure that you paste the pure path to your image file, without any extra quotes (' ' or \" \") or spaces.")
    image_path = input("Please enter the full path to your image file (e.g., /users/anay/radargram.png): ")

# Read, determine
    try:
        # Enable binary mode
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    
        # Determine type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in ('.jpg', '.jpeg'):
            mime_type = 'image/jpeg'
        elif file_extension == '.png':
            mime_type = 'image/png'
        else:
            # Default to JPEG if not PNG or JPEG, or if the extension is unusual
            print(f"Warning: Unknown file type '{file_extension}'. Using image/jpeg as default MIME type.")
            mime_type = 'image/jpeg'

    except FileNotFoundError:
        print(f"\nError: The file was not found at '{image_path}'. Please check the path and try again.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred while reading the file: {e}")
        return

# --- 4. Define the Detailed Prompt ---
    gpr_prompt = (
        "Analyze this image in detail. If it is a Ground-Penetrating Radar (GPR) radargram, "
        "identify any clear hyperbolic reflections, their relative depth/location, and "
        "suggest the potential subsurface objects or features (e.g., rebar, pipe, void). "
        "If it is not a GPR image, simply describe its contents."
    )

    # --- 5. Generate Content ---
    print("\n--- Sending Request to Gemini API... ---")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type, 
            ),
            gpr_prompt
        ]
    )

    # --- 6. Print Result ---
    print("\n====================================")
    print("       ✨ GEMINI ANALYSIS RESULT ✨       ")
    print("====================================")
    print(response.text)
    print("====================================")

def process_gpr_image(file_path):
    """
    Reads and processes the image data from the given path.
    The image is converted to a 2D intensity array (grayscale) suitable 
    for GPR analysis.
    """
    if not os.path.exists(file_path):
        print(f"{Fore.RED}❌ Error: File not found at path: {file_path}{Style.RESET_ALL}")
        return None
        
    try:
        # 1. Read the image into a NumPy array
        img_data = plt.imread(file_path)
        
        print(f"{Fore.GREEN}✅ Image loaded successfully from: {os.path.basename(file_path)}{Style.RESET_ALL}")
        print(f"Shape of the original data: {img_data.shape}")
        
        # 2. Pre-processing: Convert to Grayscale (Intensity)
        
        # Drop the alpha channel if present (4 channels -> 3 channels)
        if img_data.ndim == 3 and img_data.shape[2] == 4:
            img_data = img_data[:, :, :3]
            
        # Convert to Grayscale if it's a color image (3 channels -> 1 channel)
        if img_data.ndim == 3:
            # Standard luminance formula for grayscale conversion
            # Resulting array is 2D (Depth vs. Distance)
            gray_data = np.dot(img_data[...,:3], [0.2989, 0.5870, 0.1140])
            print("Converted image to Grayscale (2D array) for processing.")
            return gray_data
        
        # If it was already 1 channel (grayscale), return it directly
        return img_data
        

    except Exception as e:
        print(f"{Fore.RED}❌ An error occurred while reading the file: {e}{Style.RESET_ALL}")
        return None

# --- Main Script Loop ---

def gpr_reader_cli_run():
    """Main command-line interface for the GPR reader."""
    gpr_array = None
    
    print("Welcome to the GPR Image Reader.")
    print("Type 'upload <file_path>' to load an image, or 'exit' to quit.")
    print("Make sure that the name of the file does not contain spaces.")
    print("\n💡 **Examples:**")
    print("   Windows: upload C:\\Data\\profile.png")
    print("   Linux/macOS: upload /home/user/data/profile.png")
    
    while True:
        user_input = input("\n> ").strip()
        
        # 1. Handle the 'exit' command
        if user_input.lower() == 'exit':
            print("Exiting GPR Reader. Goodbye!")
            break
            
        # 2. Handle the 'upload <file_path>' command
        if user_input.lower().startswith('upload '):
            # Split the input into the command and the path
            parts = user_input.split(maxsplit=1)
            
            if len(parts) < 2:
                print("⚠️ Please provide the full path after 'upload'.")
                continue
            
            # Remove any extra quotes the user might have included
            file_path = parts[1].strip().replace('"', '').replace("'", '')
            
            # Call the processing function
            gpr_array = process_gpr_image(file_path)
            
            if gpr_array is not None:
                print("\n**Image successfully loaded and processed.**")
                
                # Show the result for confirmation
                plt.figure()
                plt.imshow(gpr_array, cmap='gray', aspect='auto')
                plt.title("Loaded GPR Profile (Intensity)")
                plt.xlabel("Distance Axis (Pixels)")
                plt.ylabel("Depth/Time Axis (Pixels)")
                plt.colorbar(label='Amplitude/Intensity')
                plt.show()
                





def print_ascii_art():
    os.system('cls' if os.name == 'nt' else 'clear')
    logo = r"""
  ____ ____  ____    _   _       _         
 / ___|  _ \|  _ \  | | | |_   _| |__      
| |  _| |_) | |_) | | |_| | | | | '_ \ 
| |_| |  __/|  _ <  |  _  | |_| | |_) |  Python CLI Edition
 \____|_|   |_| \_\ |_| |_|\__,_|_.__/ 
                                                        
    """
    cinetext_type(logo, 0.005)
    os.system('cls' if os.name == 'nt' else 'clear')
    cinetext_rainbow(logo, 50, 0.075)
    os.system('cls' if os.name == 'nt' else 'clear')
    cinetext_pulse(logo, 2, 0.05)
    time.sleep(0.5)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Fore.GREEN}{logo}{Style.RESET_ALL}")
    time.sleep(2)   
    text = (f"GPR Hub (CLI) Python edition")
    cinetext_type(text, 0.01)
    time.sleep(1)
    text = (f"Version: {version}")
    cinetext_type(text, 0.005)
    time.sleep(1)
    text = (f"Ensure that your terminal is in fullscreen. ")
    cinetext_pulse(text, 3, 0.05)
    print (" ")
    time.sleep(0.5)
    text = (f"{Style.RESET_ALL}If you face any issues, seek help from the GitHub repository:")
    cinetext_type(text, 0.005)
    text = (f"{Fore.BLUE}https://github.com/Codemaster-AR/GPR-Hub-Python{Style.RESET_ALL}.")
    cinetext_type(text, 0.005)
    text = (f"{Style.RESET_ALL}Or check the website: {Fore.BLUE}https://codemaster-ar.github.io/GPR-Hub-Python/")
    time.sleep(0.5)
    text = (f"You can also contact {Fore.BLUE}codemaster.ar@gmail.com {Style.RESET_ALL}for more details or troubleshooting.")
    cinetext_type(text, 0.005)
    text = ("")
 


def loading_bar(total_seconds=1):
    bar_length = 40
    total_items = 100
    print(f"{Fore.MAGENTA}\nProgress:")
    for i in range(total_items + 1):
        percent = 100 * i / total_items
        filled_length = int(bar_length * i // total_items)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r|{bar}| {percent:5.1f}% Complete", end='', flush=True)
        time.sleep(total_seconds / total_items)
    print(f"{Style.RESET_ALL}\n")

def start_chat_groq():
    global GROQ_API_KEY

    if not GROQ_API_KEY or "your_key" in GROQ_API_KEY:
        print("\033[1;33mWarning:\033[0m Groq API Key is not set in environment variable or hardcoded.")
        try:
            key_input = getpass("Please enter your Groq API Key (input is hidden): ")
            if key_input:
                GROQ_API_KEY = key_input
            else:
                print("\033[1;31mError:\033[0m API Key is required to start the chat.")
                return
        except Exception as e:
            print(f"\033[1;31mError during key input:\033[0m {e}")
            return

    print("-" * 52)
    print("Groq Llama3 AI Chat Initialized.")
    print("Type 'exit' or 'quit' to return to the main menu.")
    print("-" * 52)

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    while True:
        try:
            user_message = input("\033[1;32mYou:\033[0m ")
        except EOFError:
            print("\nExiting chat...")
            break
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

        if user_message.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        if not user_message.strip():
            continue

        print("Thinking...")

        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": user_message}]
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = response.read().decode('utf-8')
                data = json.loads(resp_data)
        except urllib.error.HTTPError as e:
            try:
                error_data = e.read().decode('utf-8')
                error_json = json.loads(error_data)
                error_msg = error_json.get('error', {}).get('message', str(e))
            except Exception:
                error_msg = str(e)
            print(f"\033[1;31mAPI Error:\033[0m\n{error_msg}")
            continue
        except Exception as e:
            print(f"\033[1;31mNetwork/Request Error:\033[0m {e}")
            continue

        ai_reply = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        if ai_reply:
            print("\033[1;36mGroq:\033[0m")
            try:
                cols = os.get_terminal_size().columns
            except OSError:
                cols = 80
            width = max(20, cols - 2)
            wrapped_text = textwrap.fill(ai_reply, width=width, subsequent_indent='  ')
            print(wrapped_text)
            print()
        else:
            print("\033[1;31mError:\033[0m Received empty reply from API.")
            print(f"Raw Output: {data}")

def start_chat_gemini():
    global GEMINI_API_KEY

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\033[1;33mWarning:\033[0m Gemini API Key is not set in environment variable or hardcoded.")
        try:
            key_input = getpass("Please enter your Gemini API Key (input is hidden): ")
            if key_input:
                GEMINI_API_KEY = key_input
            else:
                print("\033[1;31mError:\033[0m API Key is required to start the chat.")
                return
        except Exception as e:
            print(f"\033[1;31mError during key input:\033[0m {e}")
            return

    print("--------------------------------")
    print("Google Gemini AI Chat Initialized.")
    print("Type 'exit' or 'quit' to return to the main menu.")
    print("--------------------------------")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    while True:
        try:
            user_message = input("\033[1;32mYou:\033[0m ")
        except EOFError:
            print("\nExiting chat...")
            break
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break

        if user_message.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        if not user_message.strip():
            continue

        print("Thinking...")

        payload = {
            "contents": [{"parts": [{"text": user_message}]}],
            "systemInstruction": {
                "parts": [{
                    "text": "You are a helpful, brief, and knowledgeable assistant for Ground Penetrating Radar (GPR) analysis. Provide concise answers. Only provide information on GPRs."
                }]
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = response.read().decode('utf-8')
                data = json.loads(resp_data)
        except urllib.error.HTTPError as e:
            try:
                error_data = e.read().decode('utf-8')
                error_json = json.loads(error_data)
                error_msg = error_json.get('error', {}).get('message', str(e))
            except Exception:
                error_msg = str(e)
            print(f"\033[1;31mAPI Error:\033[0m\n{error_msg}")
            continue
        except Exception as e:
            print(f"\033[1;31mNetwork/Request Error:\033[0m {e}")
            continue

        # Gemini's response structure
        candidate = (data.get("candidates") or [{}])[0]
        ai_reply = ""
        if candidate:
            ai_reply = candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        if ai_reply:
            print("\033[1;36mGemini:\033[0m")
            try:
                cols = os.get_terminal_size().columns
            except OSError:
                cols = 80
            width = max(20, cols - 2)
            wrapped_text = textwrap.fill(ai_reply, width=width, subsequent_indent='  ')
            print(wrapped_text)
            print()
        else:
            print("\033[1;31mError:\033[0m Received empty reply from API.")
            print(f"Raw Output: {data}")

# --- Main Menu Loop ---
def main():
    gate = KeyboardGate()
    gate.KeyboardGateDisable()
    print_ascii_art()
    loading_bar(total_seconds=1)
    check_for_updates(version)
    gate.KeyboardGateEnable()
    while True:
        try:
            user_input_terminal = ("")
            user_input_terminal = input("Enter 'commands' to obtain functional commands (or Ctrl+C to stop): ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting GPR Reader. Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\nExiting GPR Reader. Goodbye!")
            sys.exit(0)

        if user_input_terminal in ["commands", "command", "cmds", "cmd", "options", "option", "features", "feature", "show commands"]:
            gate.KeyboardGateDisable()
            text = (f"\n{Style.BRIGHT}Available Commands: {Style.NORMAL}")
            cinetext_type(text, 0.005)
            text = (f"{Fore.GREEN}about_gpr{Style.RESET_ALL}      - Learn about Ground-Penetrating Radars (GPRs).")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}open_gpr{Style.RESET_ALL}       - Open the GPR image file in a graph format.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}gemini_gpr{Style.RESET_ALL}     - Allow gemini to see the GPR image and analyze it.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}read_gpr{Style.RESET_ALL}       - Read and process GPR files.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}exit{Style.RESET_ALL}           - Exit the GPR Reader Python edition.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}commands{Style.RESET_ALL}       - Display this message with available commands.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}chat groq{Style.RESET_ALL}      - Chat with Groq AI")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}chat gemini{Style.RESET_ALL}    - Chat with Google Gemini AI")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}gui_ml_gpr{Style.RESET_ALL}     - Opens the GUI website for a machine learning based GPR determiner.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.RED}text_ml_gpr{Style.RESET_ALL}    - Opens the text based machine learning gpr determiner right here.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}help{Style.RESET_ALL}           - Helps you to overcome problems you are facing with this CLI.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}version{Style.RESET_ALL}        - Show version information.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}clear{Style.RESET_ALL}          - Clear the terminal screen.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}restart{Style.RESET_ALL}        - Restart this application from the beginning.")
            cinetext_type(text, 0.0005)
            text = (f"{Fore.GREEN}github{Style.RESET_ALL}         - Open the GitHub repository for GPR Hub.")
            cinetext_type(text, 0.0005)
            text = ("Enter a command to get started. \nCommands are case sensitive.")
            cinetext_type(text, 0.0000005)
            time.sleep(0.5)
            gate.KeyboardGateEnable()
            print("──────────────────────────────────────────\n")

        elif user_input_terminal == "chat groq":
            start_chat_groq()
        
        elif user_input_terminal == "chat gemini":
            start_chat_gemini()

        elif user_input_terminal == "exit":
            print("Exiting GPR Reader. Goodbye!")
            sys.exit(0)

        elif user_input_terminal in ["analyze_data", "export_results"]:
            print(f"'{user_input_terminal}' is not implemented yet in this Python script.")

        elif user_input_terminal == "version":
            print ("\n")
            print(f"GPR Hub Python edition - Version {version}")
            text = (f"{Style.BRIGHT}Changelog:{Style.NORMAL} ")
            cinetext_type(text, 0.0005)
            text = ("Renamed from 'GPR Reader' to 'GPR Hub'")
            cinetext_type(text, 0.0005)
            text = ("Fixed AI API keys")
            cinetext_type(text, 0.0005)
            text = ("Upgraded interface with colours and effects.")
            cinetext_type(text, 0.0005)
            print("──────────────────────────────────────────\n")

        elif user_input_terminal in ["import", "imports", "library", "libraries"]:
            text = (f"{Style.BRIGHT}Python imports:")
            cinetext_type(text, 0.005)
            text = ("1. os")
            cinetext_type(text, 0.0005)
            text = ("2. sys")
            cinetext_type(text, 0.0005)
            text = ("3. time")
            cinetext_type(text, 0.0005)
            text = ("4. json")
            cinetext_type(text, 0.0005)
            text = ("5. textwrap")
            cinetext_type(text, 0.0005)
            text = ("6. urllib")
            cinetext_type(text, 0.0005)
            text = ("7. urllib.request")
            cinetext_type(text, 0.0005)
            text = ("8. urllib.error")
            cinetext_type(text, 0.0005)
            text = ("9. getpass")
            cinetext_type(text, 0.0005)
            text = ("10. matplotlib")
            cinetext_type(text, 0.0005)
            text = ("11. numpy")
            cinetext_type(text, 0.0005)
            text = ("12. genai")
            cinetext_type(text, 0.0005)
            text = ("13. google")
            cinetext_type(text, 0.0005)
            text = ("14. colorama")
            cinetext_type(text, 0.0005)
            text = ("15. cinetext")
            cinetext_type(text, 0.0005)
            text = ("16. webbrowser")
            cinetext_type(text, 0.0005)
            text = ("17. keyboard")
            cinetext_type(text, 0.0005)
            text = ("18. KeyboardGate")
            cinetext_type(text, 0.0005)
            text = ("19. ez_background_music")
            cinetext_type(text, 0.0005)
            print("──────────────────────────────────────────\n")

        elif user_input_terminal in ["help", "troubleshoot", "error", "errors"]:
            print (f"Report any errors to codemaster.ar@gmail.com or open an issue in the CLI Code Github repository ({Fore.BLUE}https://github.com/Codemaster-AR/GPR-Hub-CLI{Fore.RESET})")
     

        elif user_input_terminal == "open_gpr":
            gpr_reader_cli_run()
            break
        elif user_input_terminal == "gemini_gpr":
            gemini_image_reader()
        elif user_input_terminal == "clear":
            clear_screen()
        elif user_input_terminal == "gui_ml_gpr":
            print(f"Opening the ML GPR Analyzer website in your default browser. - Opening {Fore.BLUE}https://codemaster-ar.github.io/gpr-hub-web/ai-gpr-determiner/{Fore.RESET}...")
            openweb("https://codemaster-ar.github.io/gpr-hub-web/ai-gpr-determiner/")
        elif user_input_terminal == "text_ml_gpr":
            print("Feature under development - coming soon! Try the GUI version meanwhile by entering the 'gui_ml_gpr' command!")
        elif user_input_terminal == "about_gpr":
            print ("GPR are powerful tools that scan the underground without contact, hence mapping it without the risk of damaging the enviorment, or possibly, any artifacts.")
            print (f"Open {Fore.BLUE}")
        elif user_input_terminal in ["intro", "restart intro", "start intro", "restart"]:
            print ("This will clear the entire screen. Proceed? (y/n)")
            proceed_input = input().strip().lower()
            if proceed_input != 'y':
                print("Intro cancelled.")
                continue
            elif proceed_input in ['y', 'yes', 'proceed', 'continue']:
                clear_screen()
                print_ascii_art()
            elif proceed_input in ['n', 'no', 'cancel']:
                print("Restarting halted.")
            else:
                print("Invalid input. Please enter 'y' to proceed with restarting intro or 'n' for cancel.")
        elif user_input_terminal in ['github', 'github repository', 'repo', 'github repo']:
            text = (f"Opening the main GPR Hub Python GitHub repository ({Fore.BLUE}https://github.com/Codemaster-AR/GPR-Hub-Python{Fore.RESET}) in your default browser...")
            cinetext_type(text, 0.005)
            openweb("https://github.com/Codemaster-AR/GPR-Hub-Python")
            

        else:
            print(f"Invalid input \"{user_input_terminal}\". Please enter 'commands' to see available commands.")

        print()

def run():
    """Entry point for the console script."""
    pygame.mixer.init()
    sound_file_path = get_resource_path("Incredulity-chosic.com_.mp3")
    try:
        if os.path.exists(sound_file_path):
            pygame.mixer.music.load(sound_file_path)
            pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Warning: Could not play music: {e}")

    time.sleep(0.5)
    clear_screen()
    time.sleep(0.5)
    main()

if __name__ == "__main__":
    run()
    


 