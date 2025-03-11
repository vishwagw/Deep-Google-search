# using nltk library instead of Deep learning model:
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import re
import pyttsx3
import logging
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceSearchSummarizer:
    def __init__(self):
        # Initialize the speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # User agent to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        
        logger.info("Voice Search Summarizer initialized successfully")

    def listen_for_command(self):
        """Listen for voice command from the user"""
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

    def extract_search_query(self, command):
        """Extract the search query from the voice command"""
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

    def sentence_similarity(self, sent1, sent2, stopwords=None):
        """Calculate similarity between two sentences"""
        if stopwords is None:
            stopwords = []
        
        sent1 = [word.lower() for word in sent1.split() if word.lower() not in stopwords]
        sent2 = [word.lower() for word in sent2.split() if word.lower() not in stopwords]
        
        # Create a list of all unique words
        all_words = list(set(sent1 + sent2))
        
        # Create vectors
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # Build vectors
        for word in sent1:
            if word in all_words:
                vector1[all_words.index(word)] += 1
        
        for word in sent2:
            if word in all_words:
                vector2[all_words.index(word)] += 1
        
        # Handle empty vectors
        if sum(vector1) == 0 or sum(vector2) == 0:
            return 0.0
        
        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences, stop_words):
        """Create similarity matrix among all sentences"""
        # Initialize similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(
                        sentences[i], sentences[j], stop_words)
        
        return similarity_matrix

    def generate_summary(self, content, max_sentences=5):
        """Generate a summary of the content using TextRank algorithm"""
        if not content or len(content) < 50:
            return "I couldn't find enough information on that topic."
        
        try:
            logger.info("Generating summary...")
            
            # Get stop words
            stop_words = stopwords.words('english')
            
            # Tokenize into sentences
            sentences = sent_tokenize(content)
            
            # Limit to a reasonable number of sentences for processing
            if len(sentences) > 100:
                sentences = sentences[:100]
            
            # Create similarity matrix
            sentence_similarity_matrix = self.build_similarity_matrix(sentences, stop_words)
            
            # Apply PageRank algorithm
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
            scores = nx.pagerank(sentence_similarity_graph)
            
            # Sort sentences by score and get top n
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
            
            # Get top sentences (up to max_sentences)
            top_sentences = [ranked_sentences[i][2] for i in range(min(max_sentences, len(ranked_sentences)))]
            
            # Re-order sentences based on their original position
            ordered_sentences = sorted([(i, s) for i, (_, _, s) in enumerate(ranked_sentences[:max_sentences])], 
                                       key=lambda x: sentences.index(x[1]))
            
            summary = " ".join([s for _, s in ordered_sentences])
            
            logger.info(f"Summary generated: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "I encountered an error while generating the summary. " + str(e)

    def speak_summary(self, summary):
        """Speak the summary using text-to-speech"""
        try:
            logger.info("Converting summary to speech...")
            self.tts_engine.say(summary)
            self.tts_engine.runAndWait()
            logger.info("Text-to-speech completed")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")

    def process_request(self):
        """Process a complete voice search request"""
        # Listen for command
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

def main():
    search_summarizer = VoiceSearchSummarizer()
    
    print("Voice Search Summarizer is ready. Speak your search query...")
    while True:
        try:
            result = search_summarizer.process_request()
            print(f"Summary: {result}")
            print("\nReady for next query... (Press Ctrl+C to exit)")
        except KeyboardInterrupt:
            print("\nExiting Voice Search Summarizer")
            break

if __name__ == "__main__":
    main()
