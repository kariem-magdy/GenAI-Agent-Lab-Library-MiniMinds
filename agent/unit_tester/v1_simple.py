from ..base import Agent, BaseAgentState, LLMClient, ToolRegistry
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from pathlib import Path


class SimpleUnitTesterAgent(Agent):
    def __init__(self, llm: LLMClient,  max_iterations: int = 100):
        # TODO: create tool_registery with what you want
        tool_registry = ToolRegistry
        tool_registry.register_from_module(code_tools)
        tool_registry.register_from_module(file_tools)
        tool_registry.register_from_module(json_tools)
        

        super().__init__(llm, tool_registry, max_iterations)
        
        # TODO: inialize state
        # TODO 1: read prompts/unit_tester_v1.txt
        # TODO 2: format system prompt with tools
        # TODO 3: add system message to inital_state
        self.inital_state = BaseAgentState()
        
        # 1: Read System Prompt
        prompt_path = Path("prompts/unit_tester_v1.txt")
        if prompt_path.exists():
            sys_template = prompt_path.read_text(encoding="utf-8")
        else:
            sys_template = "You are a QA Agent. Tools: {tools}"
            
        # 2: Format Prompt with Tools
        formatted_prompt = sys_template.replace("{tools}", self.tool_registry.to_string())
        
        # 3: Add system message
        self.inital_state.add_message(role="system", content=formatted_prompt)
    
    def start_point(self, user_query) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        state = self.inital_state
        state.add_message(role="user", content=user_query)

        return state
    
    def run(self, state: BaseAgentState) -> BaseAgentState:
        # TODO 1) Call LLM
        response = self.llm_generate(state)


        # TODO 2) Add assistant response use state.add_message
        state.add_message(role="assistant", content=response.get("content"), tool_calls=response.get("tool_calls"))

        # TODO 3) Set Stop condition
        content = response.get("content") or ""
        if "finished" in content.lower() and "message" in content.lower():
             state.is_finished = True

        # TODO 4) Execute tool calls if any
        tool_calls = response.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tool_msg = self.call_tool(tc)
                state.messages.append(tool_msg)
        
        # TODO 5) return what ?
        return state
