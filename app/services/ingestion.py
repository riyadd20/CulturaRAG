"""
CulturaRAG — Document Ingestion Service
Handles chunking of raw text / PDFs / DOCX files before FAISS indexing.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import re

from loguru import logger
from app.core.config import get_settings
from app.services.vector_store import get_vector_store

settings = get_settings()


def _split_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Simple recursive character splitter.
    Priority order of split points: paragraph → sentence → word.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to split on paragraph boundary
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1:
            # Fall back to sentence boundary
            split_at = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
            )
        if split_at == -1 or split_at <= start:
            # Fall back to word boundary
            split_at = text.rfind(" ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end  # Hard cut

        chunks.append(text[start : split_at + 1].strip())
        start = split_at + 1 - chunk_overlap  # Overlap window
        start = max(start, 0)

    return [c for c in chunks if len(c) > 20]


class IngestionService:
    """Orchestrates text extraction → chunking → vector store insertion."""

    def __init__(self):
        self.vector_store = get_vector_store()

    def ingest_text(
        self,
        text: str,
        source: str,
        culture: Optional[str] = None,
        language: Optional[str] = "en",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ingest raw text string."""
        chunks_text = _split_text(text)
        chunks = [
            {
                "content": chunk,
                "source": source,
                "culture": culture,
                "language": language,
                "tags": tags or [],
                "timestamp": datetime.utcnow().isoformat(),
            }
            for chunk in chunks_text
        ]
        added = self.vector_store.add_chunks(chunks)
        logger.info(f"Ingested '{source}': {added} chunks")
        return {"chunks_added": added, "source": source}

    def ingest_pdf(
        self,
        file_bytes: bytes,
        source: str,
        culture: Optional[str] = None,
        language: Optional[str] = "en",
    ) -> Dict[str, Any]:
        """Extract text from a PDF and ingest."""
        try:
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            full_text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            return self.ingest_text(full_text, source, culture, language)
        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
            raise

    def ingest_docx(
        self,
        file_bytes: bytes,
        source: str,
        culture: Optional[str] = None,
        language: Optional[str] = "en",
    ) -> Dict[str, Any]:
        """Extract text from a DOCX and ingest."""
        try:
            import docx, io
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return self.ingest_text(full_text, source, culture, language)
        except Exception as e:
            logger.error(f"DOCX ingestion failed: {e}")
            raise

    def ingest_sample_cultural_data(self) -> int:
        """
        Seed the index with built-in sample data covering diverse cultures.
        Call once during app startup if the index is empty.
        """
        sample_docs = [
            {
                "text": (
                    "Diwali, the Festival of Lights, is one of the most significant Hindu festivals celebrated "
                    "across India and the global Indian diaspora. It spans five days, with the main celebration "
                    "on the darkest night of the Hindu lunisolar month Kartika. Homes are decorated with diyas "
                    "(oil lamps), rangoli (colorful patterns), and fireworks are lit to symbolize the victory of "
                    "light over darkness and knowledge over ignorance. Families exchange sweets, pray to Lakshmi "
                    "the goddess of prosperity, and light lamps to invite her blessings. Regional customs vary: "
                    "in West Bengal the festival coincides with Kali Puja, while Jains celebrate it as the "
                    "anniversary of Mahavira's attainment of nirvana."
                ),
                "source": "CulturaRAG Internal — Hindu Festivals",
                "culture": "Indian",
                "language": "en",
            },
            {
                "text": (
                    "Hanami (花見), literally 'flower viewing', is a centuries-old Japanese tradition of "
                    "celebrating the transient beauty of cherry blossoms (sakura). The practice dates back to "
                    "the Nara period (710–794 CE) when plum blossoms were originally admired; cherry blossoms "
                    "gained prominence during the Heian period. Today, families and friends gather in parks "
                    "under blooming sakura trees for picnics with bento boxes, sake, and traditional songs. "
                    "The Japan Meteorological Corporation issues annual sakura forecasts tracking the 'sakura "
                    "front' moving northward from Kyushu. The ephemeral nature of the blossoms—lasting only "
                    "one to two weeks—embodies the Buddhist concept of mono no aware, the poignant awareness "
                    "of impermanence."
                ),
                "source": "CulturaRAG Internal — Japanese Traditions",
                "culture": "Japanese",
                "language": "en",
            },
            {
                "text": (
                    "El Día de los Muertos (Day of the Dead) is a Mexican holiday observed on November 1–2, "
                    "blending pre-Columbian Aztec traditions with Spanish Catholic influences following the "
                    "16th-century conquest. Families build ofrendas (altars) adorned with marigold flowers "
                    "(cempasúchil), photographs of the deceased, candles, food, and personal belongings to "
                    "welcome the spirits back. The marigold's strong scent is believed to guide souls from "
                    "the land of the dead. Sugar skulls (calaveras) and skeleton imagery celebrate rather "
                    "than mourn death. In 2008, UNESCO inscribed the holiday on its list of Intangible "
                    "Cultural Heritage of Humanity. The celebration differs by region: Oaxaca features "
                    "elaborate cemetery vigils, while Michoacán is known for candlelit boat processions "
                    "on Lake Pátzcuaro."
                ),
                "source": "CulturaRAG Internal — Mexican Traditions",
                "culture": "Mexican",
                "language": "en",
            },
            {
                "text": (
                    "Ramadan is the ninth month of the Islamic lunar calendar and the holiest month for "
                    "Muslims worldwide. Muslims fast from dawn (Fajr) to sunset (Maghrib), abstaining from "
                    "food, drink, smoking, and marital relations as an act of worship and self-discipline. "
                    "The fast is broken each evening with iftar, traditionally starting with dates and water, "
                    "followed by a communal meal. The pre-dawn meal is called suhoor. The Quran was revealed "
                    "during Ramadan, making Laylat al-Qadr (Night of Power) in the final ten nights especially "
                    "sacred. Eid al-Fitr marks the end of Ramadan with prayers, feasts, charity (Zakat "
                    "al-Fitr), and gift-giving. Ramadan practices vary culturally: Moroccan families enjoy "
                    "harira soup at iftar, while Indonesians celebrate with takbiran processions."
                ),
                "source": "CulturaRAG Internal — Islamic Traditions",
                "culture": "Islamic",
                "language": "en",
            },
            {
                "text": (
                    "Ubuntu is a Nguni Bantu philosophical concept widely embraced across sub-Saharan Africa, "
                    "often translated as 'I am because we are.' It expresses the belief that a person's "
                    "humanity is realized through their relationships and interactions with others. In practice, "
                    "ubuntu manifests as communal support systems, restorative justice practices, and collective "
                    "decision-making through councils like the indaba. Archbishop Desmond Tutu popularized the "
                    "concept globally: 'A person with ubuntu is open and available to others.' The philosophy "
                    "has influenced post-apartheid South Africa's Truth and Reconciliation Commission and "
                    "is increasingly cited in discussions of African leadership, economic cooperation, and "
                    "technology ethics—inspiring the name of the Ubuntu Linux operating system."
                ),
                "source": "CulturaRAG Internal — African Philosophy",
                "culture": "African",
                "language": "en",
            },
            {
                "text": (
                    "中国春节 (Chūnjié), the Chinese New Year or Spring Festival, is the most important "
                    "traditional holiday in Chinese culture. Based on the lunisolar Chinese calendar, it "
                    "falls between January 21 and February 20. The 15-day celebration begins on New Year's "
                    "Eve with reunion dinners (年夜饭, niányèfàn), where families share symbolic foods: fish "
                    "(鱼, yú) for abundance, dumplings (饺子, jiǎozi) shaped like gold ingots for wealth, "
                    "and tangyuan for togetherness. Red envelopes (红包, hóngbāo) filled with money are "
                    "given to children and unmarried adults. Dragon dances, lion dances, and fireworks drive "
                    "away evil spirits. Lantern Festival on the 15th day marks the end with riddle-solving "
                    "and lantern displays. The festival is observed by Chinese communities globally and "
                    "influences markets through the 'Golden Week' shopping surge."
                ),
                "source": "CulturaRAG Internal — Chinese Traditions",
                "culture": "Chinese",
                "language": "en",
            },
        ]

        total = 0
        for doc in sample_docs:
            result = self.ingest_text(**doc)
            total += result["chunks_added"]
        logger.info(f"Seeded {total} sample cultural chunks into the index.")
        return total


def get_ingestion_service() -> IngestionService:
    return IngestionService()
