import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import warnings
import faiss
from gtts import gTTS
import time

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class AIVoiceAssistant:
    def __init__(self):
        # Get Hugging Face API token from environment variable
        hf_api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not hf_api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")
        
        # Initialize Mistral 7B model using HuggingFaceInferenceAPI
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self._llm = HuggingFaceInferenceAPI(
            model_name=model_id,
            token=hf_api_token,
            task="text-generation",
            max_new_tokens=256,
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.95,
            },
        )
        
        # Initialize embedding model
        self._embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Update global settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model
        
        self._index = None
        if not self._create_kb():
            raise RuntimeError("Failed to create knowledge base")
        self._create_chat_engine()

        # Initialize audio output directory
        self.audio_output_dir = "audio_output"
        os.makedirs(self.audio_output_dir, exist_ok=True)

    def _create_chat_engine(self):
        if self._index is None:
            raise ValueError("Index has not been created. Cannot create chat engine.")
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            # Use os.path.join for cross-platform compatibility
            file_path = os.path.join(os.getcwd(), "rag", "restaurant_file.txt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            
            # Create a FAISS index
            d = 384  # dimensionality of the embedding model
            faiss_index = faiss.IndexFlatL2(d)
            
            # Create a FaissVectorStore
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Create a storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create the index
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
            return True
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            return False

    def interact_with_llm(self, customer_query):
        if self._chat_engine is None:
            raise ValueError("Chat engine has not been initialized")
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer

    def text_to_speech(self, text, language='en', slow=False):
        tts = gTTS(text=text, lang=language, slow=slow)
        
        timestamp = int(time.time())
        audio_file = f"{self.audio_output_dir}/response_{timestamp}.mp3"
        tts.save(audio_file)
        
        print(f"Audio response saved to: {audio_file}")
        return audio_file

    def process_voice_input(self, transcription):
        # Process customer input and get response from AI assistant
        response = self.interact_with_llm(transcription)
        
        # Convert the response to speech and save as audio file
        audio_file = self.text_to_speech(response)
        
        return f"Customer: {transcription}\nAI Assistant: {response.lstrip()}\nAudio Response: {audio_file}"

    @property
    def _prompt(self):
        return """
            You are a professional AI Assistant receptionist working in Bangalore's one of the best restaurant called Bangalore Kitchen,
            Ask questions mentioned inside square brackets which you have to ask from customer, DON'T ASK THESE QUESTIONS 
            IN ONE go and keep the conversation engaging ! always ask question one by one only! 
            
            [Ask Name and contact number, what they want to order and end the conversation with greetings!] 
            
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Provide concise and short answers not more than 10 words, and don't chat with yourself!
            """