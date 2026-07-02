from langchain_text_splitters import RecursiveCharacterTextSplitter

trial_text = """
Marine animals are fascinating creatures that live in oceans, seas, and other saltwater environments around the world. They come in a wide variety of shapes, sizes, and colors, ranging from tiny plankton to enormous blue whales, the largest animals on Earth. Marine ecosystems are home to fish, dolphins, sharks, sea turtles, octopuses, jellyfish, crabs, and countless other species. These animals play important roles in maintaining the balance of ocean life. For example, sharks help regulate fish populations, while coral reefs provide shelter and breeding grounds for many marine organisms. Marine mammals such as whales and dolphins are known for their intelligence and complex communication systems. However, marine animals face numerous threats, including pollution, overfishing, climate change, and habitat destruction. Plastic waste in the oceans can harm or kill many species when ingested. Conservation efforts, such as marine protected areas and sustainable fishing practices, are essential to protect these animals and preserve biodiversity. By learning about and caring for marine life, humans can help ensure healthy oceans for future generations.
"""

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_text(trial_text)

print(len(chunks))
print(chunks)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks,embedding_model)
vector_store.save_local("vector_store")