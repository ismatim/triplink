# 🌍 TripLink Agent

**TripLink Agent** is an AI-powered travel concierge that transforms YouTube travel guides into professional, structured itineraries. Using **Retrieval-Augmented Generation (RAG)**, the agent doesn't just summarize a video; it builds a searchable memory of the trip, allowing you to ask follow-up questions about prices, locations, and logistics in your preferred language.

---

## ✨ Features

- **Smart Search:** Uses `yt-dlp` to automatically find the most relevant travel guide based on a user-provided location.
- **Dual-Context RAG:** The agent uses both the **generated itinerary** and the **raw transcript segments** to answer questions, ensuring it understands "Day 1" or "Day 2" references.
- **Multi-Language Support:** Automatically translates guides and answers into your target language (default: **Spanish**).
- **Persistent Memory:** Built on **FAISS** and **OpenAI 2026 Embeddings**, allowing for high-speed, local similarity searches.
- **Automated Documentation:** Exports the master plan into a clean `.md` file for offline use.
- **Cookie Integration:** Supports `youtube_cookies.txt` to bypass bot detection and access age-restricted content.

---

## 🏗️ Architecture

The app follows a modern **Agentic Workflow**:

1. **Ingestion:** Location search ➡️ YouTube Metadata ➡️ Transcript Extraction.
2. **Vectorization:** Recursive chunking ➡️ `text-embedding-3-large` ➡️ FAISS Index.
3. **Synthesis:** Full transcript analysis via `gpt-4o-mini` to create the master plan.
4. **Concierge:** Interactive loop using specialized Q&A prompts and vector retrieval.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An **OpenAI API Key** (set in your environment).
- (Optional) `youtube_cookies.txt` in Netscape format for better YouTube access.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/triplink-agent.git
   cd triplink-agent
   ```

2. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

   _Note: Ensure you have `yt-dlp`, `langchain-openai`, `faiss-cpu`, and `youtube-transcript-api` installed._

3. **Setup Environment:**
   Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_actual_key_here
   ```

---

## 🛠️ Usage

Simply run the main script and follow the interactive prompts:

```bash
python main.py
```

### Example Workflow

1. **Input:** `Rio de Janeiro travel guide`
2. **Agent:** Searches YouTube, finds a guide, and generates `trip_rio_de_janeiro.md`.
3. **Concierge:** \* _User:_ "What was the food recommendation for Day 2?"
   - _Agent:_ "Paseo por la playa de Copacabana por la mañana para disfrutar de un desayuno con vista al mar..."

---

## ⚙️ Configuration

You can customize the agent's behavior in `main.py`:

```python
# Change target language to English or Portuguese
agent = TripLinkAgent(language="English")

# Adjust chunk size for shorter/longer videos
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

---

## Roadmap

### Phase 1: Stabilization & UX

- Disk Persistence: Save/load the FAISS index locally so you don't have to re-index videos every session.
- Deep-Linking: Auto-generate YouTube links with timestamps (&t=seconds) for instant verification.
- Auto-Language Sync: Automatically match the output language to the user's input (e.g., reply in Spanish if asked in Spanish).
- Live Web Verification: Use tools like Tavily to check if restaurants mentioned in older videos are still open.

---

## 📝 License

BSD License
