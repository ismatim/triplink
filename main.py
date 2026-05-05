import os
import sys
import re
import requests
import logging
from dotenv import load_dotenv
from http.cookiejar import MozillaCookieJar
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


class TripLinkAgent:
    def __init__(
        self, api_key=None, cookies_path="youtube_cookies.txt", language="Spanish"
    ):
        """Initializes the Agent and the OpenAI 2026 intelligence stack."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cookies_path = cookies_path

        # 1. Initialize Intelligence Models
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=self.api_key
        )

        # 2. Agent State (Memory)
        self.video_id = None
        self.full_transcript = ""
        self.vector_db = None
        self.qa_prompt = self._create_travel_qa_expert_prompt()
        self.last_itinerary = ""
        self.language = language

    def _get_video_id(self, url):
        """Internal helper to extract the 11-char YouTube ID."""
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else url

    def search_and_load(self, location):
        """Uses yt-dlp to find the most relevant travel guide for a location."""
        search_query = f"ytsearch1:{location} travel guide"
        print(f"🔍 Searching YouTube for the best '{location}' guide...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_query, download=False)
                if "entries" in result and result["entries"]:
                    video_url = result["entries"][0]["url"]
                    video_title = result["entries"][0].get("title", "Unknown Title")
                    print(f"🎬 Found: {video_title}")
                    print(f"🔗 URL: {video_url}")
                    return self.load_video(video_url)

            print("❌ No videos found for that location.")
            return False
        except Exception as e:
            print(f"❌ yt-dlp Search Error: {e}")
            return False

    def load_video(self, url):
        """Fetches the transcript, formats it, and builds the FAISS memory index."""
        self.video_id = self._get_video_id(url)
        print(f"📡 Accessing Video ID: {self.video_id}...")

        # Setup Session with Cookies
        session = requests.Session()
        if os.path.exists(self.cookies_path):
            cj = MozillaCookieJar(self.cookies_path)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(cj)
        else:
            print(f"⚠️ Warning: {self.cookies_path} not found.")

        try:
            # Fetch Data
            api = YouTubeTranscriptApi(http_client=session)
            raw_data = api.fetch(self.video_id, languages=["es", "en"])

            # Format text
            self.full_transcript = "".join(
                [f"Text: {entry.text} Start: {entry.start:.2f}\n" for entry in raw_data]
            )

            # Build RAG Memory
            print("🛰️  Indexing travel segments into Vector Memory...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            chunks = splitter.split_text(self.full_transcript)
            self.vector_db = FAISS.from_texts(chunks, self.embeddings)

            print("✅ Agent Memory Online.")
            return True

        except Exception as e:
            print(f"❌ Load Error: {e}")
            return False

    def generate_itinerary(self, location):
        """Generates a comprehensive day-by-day itinerary using the full transcript."""
        if not self.full_transcript:
            return "Error: No video loaded. Please load a video first."

        template = """
        You are an elite Travel Planning Agent. Extract travel recommendations 
        from the following YouTube transcript and organize them into a day-by-day sequence for {location}.
        Include transport, budget, and pro-tips. Do not include timestamps.

        CRITICAL: You must provide the entire itinerary in {language}.

        TRANSCRIPT:
        {transcript}

        TRAVEL PLAN:
        """
        prompt = PromptTemplate(
            input_variables=["transcript", "language", "location"], template=template
        )

        # Modern LCEL Chain
        chain = prompt | self.llm | StrOutputParser()

        print("🤖 Agent is analyzing the full transcript for an itinerary...")
        self.last_itinerary = chain.invoke(
            {
                "transcript": self.full_transcript,
                "language": self.language,
                "location": location,
            }
        )
        return self.last_itinerary

    def _create_travel_qa_expert_prompt(self):
        template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are the 'TripLink Fact-Finder.' 

        You have two sources of information:
        1. THE ITINERARY: A summary of the trip already planned.
        2. TRANSCRIPT CONTEXT: Raw snippets from the video.

        INSTRUCTIONS:
        - If the user asks about "Day 1" or "Day 2," look at THE ITINERARY to see what was planned for that day, then use the TRANSCRIPT CONTEXT to provide details.
        - If the answer isn't in either, say you don't know.
        - Bold **locations** and **prices**.

        ITINERARY:
        {itinerary}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        TRANSCRIPT CONTEXT:
        {context}

        USER QUESTION:
        {query}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

        CRITICAL: Always answer the user in {language}.
        """
        return PromptTemplate(
            input_variables=["itinerary", "context", "query", "language"],
            template=template,
        )

    def ask_question(self, user_query, k=5):
        """Uses the specialized Fact-Finder prompt for RAG answers."""
        if not self.vector_db:
            return "Please load a destination first."

        # 1. Retrieve the 'Evidence' (RAG)
        # k=5 gives a good balance of detail vs. speed
        docs = self.vector_db.similarity_search(user_query, k=k)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])

        # 2. Build the specialized QA Chain
        # We add the StrOutputParser here to keep the terminal clean
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()

        # 3. Generate Answer
        print(f"🕵️  Agent is cross-referencing transcript for evidence...")
        return qa_chain.invoke(
            {
                "itinerary": self.last_itinerary,
                "context": context_text,
                "query": user_query,
                "language": self.language,
            }
        )


# --- Main Execution ---
if __name__ == "__main__":
    agent = TripLinkAgent()

    print("\n" + "🌍" * 3 + " TRIPLINK 2026 " + "🌍" * 3)
    target_location = input("📍 Where would you like to go? ").strip()

    if target_location:
        # Phase 1: Search, Load, and Plan
        if agent.search_and_load(target_location):
            print("\n" + "=" * 50)
            print("\n🌍 Welcome to TripLink Agent 2026")
            print("=" * 50)

            itinerary = agent.generate_itinerary(target_location)
            print(itinerary)

            # Save the file
            fname = f"trip_{target_location.lower().replace(' ', '_')}.md"
            with open(fname, "w") as f:
                f.write(itinerary)
            print(f"\n💾 Itinerary saved to {fname}")

            # Phase 2: Interactive Chat (RAG)
            print("\n" + "✨" * 3 + " CONCIERGE MODE ACTIVE " + "✨" * 3)
            print(
                "Ask me anything about the trip (e.g., 'What was the food recommendation?')"
            )
            print("Type 'exit' to end the session.")

            while True:
                user_query = input("\n🔎 Question: ").strip()

                if user_query.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Enjoy your trip to " + target_location + "!")
                    break

                if not user_query:
                    continue

                answer = agent.ask_question(user_query)
                print(f"\n💡 AGENT: {answer}")
                print("-" * 30)

        else:
            print("🛑 Failed to initialize trip planning.")
