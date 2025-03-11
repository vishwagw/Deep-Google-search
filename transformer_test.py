# using transformers as deep learning neural network architecture:
# building a deep google searching alogorithm for Trinity:
import speech_recognition as sr
import pyttsx3
import requests
import webbrowser
import re
import logging
from bs4 import BeautifulSoup
from transformers import pipeline
from googlesearch import search

# setting up the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# building the test assistant class:
class testAssistant:
    def __init__(self):

        # initialize the organizer:
        self.recognizer = sr.Recognizer()

        # Initialize the summarization pipeline
        logger.info("Loading summarization model...")
        self.summarizer = pipeline("summarization")
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # User agent to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        
        logger.info("Voice Search Summarizer initialized successfully")

    # function for listening to user command:
    def listen_to_command(self):
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("Listening for voice command...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                logger.info("Processing audio...")
                
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Recognized: {text}")
                
                return text
            except sr.WaitTimeoutError:
                logger.warning("Timeout waiting for voice command")
                return None
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except Exception as e:
                logger.error(f"Error in speech recognition: {str(e)}")
                return None
            
    # extracting the search query from user voice input/command:
    def extract_search_query(self, command):

        # Common patterns like "search for X", "find information about X", etc.
        patterns = [
            r"search for (.*)",
            r"search (.*)",
            r"find information about (.*)",
            r"find (.*)",
            r"tell me about (.*)",
            r"what is (.*)",
            r"who is (.*)",
            r"where is (.*)",
            r"when is (.*)",
            r"how to (.*)"
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, use the whole command as the query
        return command.strip()
    
    # deep google search:
    def search_google(self, query):
        """Search Google for the query and return relevant text content"""
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        try:
            logger.info(f"Searching Google for: {query}")
            response = requests.get(search_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from search results
            # Focus on the main content blocks that typically contain useful information
            content_blocks = []
            
            # Featured snippets
            featured_snippet = soup.select('.hgKElc')
            if featured_snippet:
                content_blocks.extend([block.get_text() for block in featured_snippet])
            
            # Knowledge panels
            knowledge_panel = soup.select('.kno-rdesc span')
            if knowledge_panel:
                content_blocks.extend([block.get_text() for block in knowledge_panel])
            
            # Regular search results
            search_results = soup.select('.VwiC3b')
            if search_results:
                content_blocks.extend([result.get_text() for result in search_results])
            
            # If we still don't have enough content, get all paragraph texts
            if len(' '.join(content_blocks)) < 200:
                paragraphs = soup.select('p')
                content_blocks.extend([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
            # Combine all content
            combined_content = ' '.join(content_blocks)
            
            # Clean up the text
            combined_content = re.sub(r'\s+', ' ', combined_content).strip()
            
            logger.info(f"Extracted {len(combined_content)} characters of content")
            return combined_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching Google: {str(e)}")
            return None
        
    # generate a summary:
    def generate_summary(self, content, max_length=150):
        if not content or len(content) < 50:
            return "I couldn't find enough information on that topic."
        
        try:
            # Limit input text to prevent issues with the model
            input_text = content[:4000]
            
            logger.info("Generating summary...")
            summary = self.summarizer(input_text, 
                                     max_length=max_length, 
                                     min_length=30, 
                                     do_sample=False)
            
            summary_text = summary[0]['summary_text']
            logger.info(f"Summary generated: {len(summary_text)} characters")
            return summary_text
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "I encountered an error while generating the summary."
        
    # speaking the summary result:
    def speak_summary(self, summary):
        try:
            logger.info("Converting summary to speech...")
            self.tts_engine.say(summary)
            self.tts_engine.runAndWait()
            logger.info("Text-to-speech completed")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")

    # processing a request:
    def process_request(self):
        command = self.listen_for_command()
        if not command:
            return "Sorry, I didn't hear a search command."
        
        # Extract search query
        query = self.extract_search_query(command)
        if not query:
            return "I couldn't understand what you want to search for."
        
        # Search Google
        content = self.search_google(query)
        if not content:
            return "I couldn't retrieve search results for that query."
        
        # Generate summary
        summary = self.generate_summary(content)
        
        # Speak the summary
        self.speak_summary(summary)
        
        return summary
    
# building the main program function:
def main():
    search_summarizer = testAssistant()
    
    print("Voice Search Summarizer is ready. Speak your search query...")
    while True:
        try:
            result = search_summarizer.process_request()
            print(f"Summary: {result}")
            print("\nReady for next query... (Press Ctrl+C to exit)")
        except KeyboardInterrupt:
            print("\nExiting Voice Search Summarizer")
            break

# initializing the test program:
if __name__ == "__main__":
    main()
