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
