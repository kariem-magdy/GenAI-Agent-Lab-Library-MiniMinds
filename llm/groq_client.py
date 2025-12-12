import os
from typing import Iterator, List, Optional
from dotenv import load_dotenv
from groq import Groq
from .base import LLMClient
from .config import LLMConfig

# TODO 1: load dotenv
load_dotenv()

class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # TODO 2: create groq client and set api_key from .env
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
             # Fallback for testing if env var not set, though ideally it should be.
             pass
        self.client = Groq(api_key=api_key)
    
    def generate(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        # TODO: write description for Returend Fields 
        """ 
        Main Returned Dict Fields:
            - role: The role of the message sender (usually "assistant").
            - content: The text content of the response.
            - tool_calls: A list of tool calls if the model decided to call tools.
        """
        # TODO 3: call `client.chat.completions.create` with configurations in self.config
        # TODO 3: search difference between max_tokens and max_compeletion_tokens
        # NOTE: max_tokens usually limits the generated tokens. max_completion_tokens is the newer field name 
        # specifically for OpenAI's O1 models, but Groq typically uses max_tokens.
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.model_dump()
    
    def stream(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Iterator[dict]:
        # TODO 3: call `client.chat.completions.create` with stream options configurations in self.config
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices:
                yield chunk.choices[0].delta.model_dump()
                
                
if __name__ == "__main__":
    #TODO: initlaize configuraiton with reasoning model -- search for groq reasoning models 
    config = LLMConfig(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024
    )
    
    try:
        client = GroqClient(config)
        
        #TODO: write messages with (1. system prompt on how the model is QA engineer and know python, playwright etc... provide in course) 
        system_prompt = """You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness."""

        #TODO: (2. Ask model to "write a plan to build a software autonomus like cursor but for testing") 
        messages = [
            {
                "role": "system", "content": system_prompt,
            },
            {
                "role": "user", "content": "write a plan to build a software autonomus like cursor but for testing"   
            }
        ]
        
        print("--- Testing Generate ---")
        #TODO: test client.generate
        response = client.generate(messages)
        print("Response Content:", response.get('content'))
        
        # Add the assistant's response to history for multi-turn
        messages.append({"role": "assistant", "content": response.get('content')})

        print("\n--- Testing Stream (and difference explanation) ---")
        #TODO: test client.stream and mention what's difference and why we need it?
        # YOUR_ANSWER: Streaming allows us to receive tokens as they are generated, reducing perceived latency.
        
        stream_message = [{"role": "user", "content": "Summarize that plan in one sentence."}]
        messages_for_stream = messages + stream_message
        
        print("Streaming Response:")
        full_streamed_content = ""
        for chunk in client.stream(messages_for_stream):
            content = chunk.get('content')
            if content:
                print(content, end="", flush=True)
                full_streamed_content += content
        print()

        #TODO add the new answer to messages -> create multi-turn conversation
        
        print("\n--- Testing Multi-turn Conversation ---")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "your name is CHATTAH tester"},        
        ]
        
        # TODO: first turn
        print("User: your name is CHATTAH tester")
        answer_1 = client.generate(messages)
        print("Assistant:", answer_1.get('content'))
        messages.append({"role": "assistant", "content": answer_1.get('content')})

        # TODO: second turn
        next_user_msg = {"role": "user", "content": "tell me what's your name and what are language you expert in it ?"}
        messages.append(next_user_msg)
        print(f"User: {next_user_msg['content']}")
        
        answer_2 = client.generate(messages)
        print("Assistant:", answer_2.get('content'))
        messages.append({"role": "assistant", "content": answer_2.get('content')})
        
    except Exception as e:
        print(f"Skipping execution because of error (likely missing API Key): {e}")