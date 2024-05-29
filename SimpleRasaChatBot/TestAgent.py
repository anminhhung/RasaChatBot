import asyncio
from rasa.core.agent import Agent

# # Load the trained model
agent = Agent.load("SimpleRasaChatBot\models")

# # Parse a sample text
# text = "Hello, how are you?"
text = "xin chào, bạn có khỏe không?"
result = asyncio.run(agent.parse_message(message_data=text))

# # # Print the features
print(result)
# # print(result["text"])
# # print(result["intent"])
# # print(result["entities"])
# # print(result["text_features"])
print("Done!")

# from Custom.BARTPHOTOKENIZER import BartPhoTokenizer

# text = "hello there"

# tokens = BartPhoTokenizer.tokenize(text)
# print(tokens)