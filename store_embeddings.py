from langchain_text_splitters import RecursiveCharacterTextSplitter

trial_text = """
Marine animals are a diverse group of organisms that inhabit the world's oceans, seas, estuaries, and coastal regions. They play a vital role in maintaining the health and balance of marine ecosystems. Marine life includes mammals, fish, reptiles, birds, invertebrates, and microscopic organisms. Oceans cover more than 70% of the Earth's surface and contain millions of species, many of which have yet to be discovered.

Marine mammals are warm-blooded animals that depend on the ocean for survival. Examples include whales, dolphins, porpoises, seals, sea lions, manatees, dugongs, and sea otters. Whales are among the most remarkable marine animals. The blue whale is the largest animal known to have ever existed, reaching lengths of up to 30 meters and weighing more than 150 tons. Dolphins are highly intelligent marine mammals known for their social behavior, problem-solving abilities, and communication skills. Many dolphin species live in groups called pods and use a variety of clicks, whistles, and body movements to communicate.

Fish represent the largest group of marine vertebrates. They come in a vast range of sizes, colors, and adaptations. Some common marine fish include tuna, cod, clownfish, angelfish, mackerel, and swordfish. Fish breathe through gills, which extract oxygen from water. Some fish, such as the clownfish, have symbiotic relationships with other marine organisms. Clownfish live among sea anemones, gaining protection from predators while helping the anemone by cleaning it and attracting prey.

Sharks are among the most well-known marine predators. They have existed for more than 400 million years and have evolved into over 500 different species. Sharks possess highly developed senses, including an excellent sense of smell and the ability to detect electrical signals produced by other animals. Contrary to popular belief, most shark species pose little threat to humans. Important species include the great white shark, whale shark, tiger shark, hammerhead shark, and bull shark. The whale shark is the largest fish in the world and feeds primarily on plankton.

Marine reptiles include sea turtles, sea snakes, marine iguanas, and saltwater crocodiles. Sea turtles are ancient creatures that have inhabited the oceans for more than 100 million years. The seven species of sea turtles include the green turtle, hawksbill turtle, leatherback turtle, loggerhead turtle, olive ridley turtle, Kemp's ridley turtle, and flatback turtle. Sea turtles migrate long distances between feeding and nesting areas. Female turtles return to beaches to lay eggs, often on the same shores where they were born.

Octopuses are among the most intelligent invertebrates in the ocean. They belong to the cephalopod group, which also includes squids and cuttlefish. Octopuses have eight arms lined with suction cups that help them grasp objects and capture prey. They are known for their problem-solving abilities, camouflage skills, and flexibility. An octopus can change the color and texture of its skin to blend into its surroundings. Some species can even mimic other marine animals as a defense mechanism.

Jellyfish are simple but fascinating marine animals. They have existed for over 500 million years and consist mostly of water. Jellyfish drift through ocean currents and use their tentacles to capture prey. Some species have mild stings, while others possess powerful toxins that can be dangerous to humans. Despite lacking a brain, jellyfish can respond effectively to environmental changes.

Coral reefs are often called the rainforests of the sea because they support an incredible diversity of marine life. Corals are tiny animals known as polyps that build hard calcium carbonate structures. Over thousands of years, these structures form coral reefs. Reefs provide food, shelter, and breeding grounds for thousands of species, including fish, crustaceans, mollusks, and sea turtles. Famous coral reef systems include the Great Barrier Reef in Australia, the largest coral reef system in the world.

Marine invertebrates make up the majority of ocean species. This group includes crabs, lobsters, shrimp, starfish, sea urchins, sea cucumbers, clams, oysters, mussels, and many others. Crustaceans such as crabs and lobsters have hard exoskeletons that protect their bodies. Echinoderms such as starfish and sea urchins possess unique radial symmetry and the ability to regenerate lost body parts. For example, some starfish can regrow an entire arm if it is damaged.

Marine birds are closely associated with ocean environments. Examples include penguins, albatrosses, gulls, pelicans, puffins, and cormorants. Penguins are flightless birds adapted for swimming and are primarily found in the Southern Hemisphere. Albatrosses are famous for their long-distance flights and can travel thousands of kilometers across oceans without landing.

The ocean is divided into several zones based on depth and sunlight availability. The sunlight zone, also known as the epipelagic zone, extends from the surface to about 200 meters deep. Most marine life is concentrated here because sunlight supports photosynthesis. The twilight zone lies between 200 and 1,000 meters, where little sunlight penetrates. Below this is the midnight zone, characterized by complete darkness, high pressure, and cold temperatures. Some deep-sea animals produce their own light through a process called bioluminescence. Examples include anglerfish, lanternfish, and certain species of squid.

Marine food chains begin with microscopic organisms known as phytoplankton. These tiny plants use sunlight to produce energy through photosynthesis and generate a significant portion of the Earth's oxygen. Zooplankton feed on phytoplankton and, in turn, become food for larger animals such as fish, whales, and jellyfish. Predators such as sharks, seals, and large fish occupy higher levels of the food chain.

Marine animals have developed numerous adaptations to survive in their environments. Camouflage allows animals to avoid predators or ambush prey. Schooling behavior helps fish reduce the risk of predation. Blubber provides insulation for marine mammals living in cold waters. Streamlined body shapes enable fast swimming and efficient movement through water. Some species migrate thousands of kilometers annually in search of food or breeding grounds.

Human activities significantly impact marine ecosystems. Plastic pollution is one of the greatest threats facing marine animals today. Millions of tons of plastic enter the oceans each year, harming wildlife through ingestion and entanglement. Overfishing reduces fish populations and disrupts food chains. Climate change causes rising ocean temperatures, coral bleaching, sea level rise, and ocean acidification. These changes affect countless marine species and their habitats.

Conservation efforts aim to protect marine biodiversity and ensure the long-term health of ocean ecosystems. Marine protected areas restrict harmful activities and provide safe habitats for wildlife. Sustainable fishing practices help prevent overexploitation of fish stocks. International organizations, governments, scientists, and local communities work together to monitor marine populations and restore damaged ecosystems.

Marine animals are essential to the health of the planet. They contribute to food security, tourism, scientific research, and ecological balance. Oceans regulate climate, absorb carbon dioxide, and produce much of the oxygen needed for life on Earth. Understanding marine animals and their habitats is crucial for promoting conservation and ensuring that future generations can continue to benefit from healthy and thriving oceans.

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
