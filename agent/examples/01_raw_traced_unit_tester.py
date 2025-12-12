"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result

Difference Between it And `00_raw_traced_unit_tester.py` no mlflow/langfuse 
"""
import json
from loguru import logger
from langfuse import observe, get_client
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from llm.config import LLMProvider
from pathlib import Path


langfuse = get_client()

@observe(name="llm-call", as_type="generation")
def traced_client_generate(client, messages, tools):
    #TODO: call client (provide tools in .generate) with registery.to_client_tools()
    return client.generate(messages, tools=tools)


@observe(name="tool-call", as_type="tool")
def traced_tool_execution(registry, tool_call):
    try:
        # TODO: extract func_name, args and call it -> set tool_message {content: result}
        func_name = tool_call["function"]["name"]
        args_raw = tool_call["function"]["arguments"]
        
        if isinstance(args_raw, str):
            func_inputs = json.loads(args_raw)
        else:
            func_inputs = args_raw
            
        tool_instance = registry.get(func_name)
        if not tool_instance:
             raise ValueError(f"Tool {func_name} not found")
             
        func_results = tool_instance(**func_inputs)

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": str(func_results),
        }
        return tool_message
        
    except Exception as error:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": str(error),
        }
                
# ================ 1. Initalization ================
# 1.1 setup llm client
config = LLMConfig(
    max_tokens=5000,
    model_name="llama-3.3-70b-versatile",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 1.2 setup path
files_under_test = ["tools/toolkit/web_explorer.py"]
tests_output_directory_path = "tools/llm_tests"

messages = [
    {
        "role": "system", "content": """
        You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness. 
        Analyze each tool's behavior, suggest improvements if needed, and provide a clear, structured test report based on your findings.
        
        Output:
            - "finished": <boolean, indicate if the task is complete>
            - "message": <summary and coverage of tests>
        
        **Use only this tools:**
        {tools}
        you will be penalized if you use OTHER TOOLS
        """
    },
    {
        "role": "user", "content": """write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {tests_output_directory_path}
        """ 
    }
]
# TODO: 1.3 create tool register and add tools/modules you need
registry = ToolRegistry()
registry.register_from_module(web_explorer_tools)
registry.register_from_module(code_tools)
registry.register_from_module(file_tools)
registry.register_from_module(json_tools)

# TODO: 1.4 add tools to system_message use str.format method and  registery.to_string()
messages[0]["content"] = messages[0]["content"].replace("{tools}", registry.to_string())

# TODO: 1.5 set `files_under_test` and `tests_output_directory_path` in user_message like .format
messages[1]["content"] = messages[1]["content"].format(
    files_under_test=str(files_under_test),
    tests_output_directory_path=str(tests_output_directory_path)
)

# TODO 1.6 set root span with root_span = langfuse.start_span(name , metadata)
root_span = langfuse.trace(name="unit-tester-run", metadata={"files": files_under_test})

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    
    # Use Langfuse span context
    with root_span.span(name=f"iteration-{iteration}"):
        # TODO 2.1 call client (provide tools in .generate) with registery.to_client_tools()
        client_tools = registry.to_client_tools(config.provider)
        response = traced_client_generate(client, messages, tools=client_tools)
        
        # TODO 2.2 append assistant message (role, content, **tool_calls**) *log it logger.info*
        messages.append(response)
        logger.info(f"Assistant: {response.get('content')}")

        # TODO get content and check if is finished
        # 2.3 Stop when one of the conditions happen
        # 2.3 'finished' in response['content'] -- handle response['content']=None case
        # 2.3 exceed max_iterations
        content = response.get("content") or ""
        
        if iteration > max_iterations:
            break
            
        finished = False
        if content:
            try:
                clean_content = content.strip()
                if clean_content.startswith("```"):
                     clean_content = clean_content.split("\n", 1)[-1].rsplit("\n", 1)[0]
                data = json.loads(clean_content)
                if data.get("finished") is True:
                    finished = True
            except json.JSONDecodeError:
                pass
        
        if finished:
            logger.success("Agent finished task.")
            break
        
        # 2.4 execute any function execturion inside `tool_calls` || handle if it's None or not passed
        tool_calls = response.get("tool_calls", []) or []
        for tool_call in tool_calls:
            if tool_call["type"] != "function":
                continue
            
            tool_message = traced_tool_execution(registry, tool_call)
                
            messages.append(tool_message)
            logger.info(f"tool response {json.dumps(tool_message, indent=2)}")
            
# Close the trace
root_span.update(output=messages[-1])