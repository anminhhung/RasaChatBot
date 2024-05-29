import asyncio
from rasa.core.agent import Agent

# # Load the trained model
# agent = Agent.load("Test\models")
agent = Agent.load("Test\models\\20240529-231644-international-road.tar.gz")

# # Parse a sample text
# text = "Hello, how are you?"
text = "Tôi tên Xuân, CCCD 789012345678 và sdt là 0835678910"
result = asyncio.run(agent.parse_message(message_data=text))

# # Print the features
print(result)
# print(result["text"])
# print(result["intent"])
# print(result["entities"])
# print(result["text_features"])
print("Done!")
