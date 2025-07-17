from autogen import ConversableAgent, Agent
from typing import Union, Any
from autogen.coding import CodeExecutor, CodeExtractor
from IPython.core.interactiveshell import InteractiveShell
from typing import Optional
from autogen.coding.base import CodeExecutionConfig, CodeBlock, CodeResult
import json
import sys
from io import StringIO
import re

class CustomCodeExecutor(CodeExecutor):
    def __init__(self, shell = None, extractor=None):
        self.shell = shell if shell is not None else InteractiveShell()
        self.execute_python = get_execute_python(self.shell)
    
    @property
    def code_extractor(self) -> CodeExtractor:
        return CustomCodeExtractor

    def save_session(self, filepath):
        """Save session using dill"""
        code_block = CodeBlock(code=f"import dill\ndill.dump_session('{filepath}')", language='python')
        return self.execute_code_blocks([code_block])

    def load_session(self, filepath):
        """Load session using dill"""
        #self.shell.reset()
        code_block = CodeBlock(code=f"import dill\ndill.load_session('{filepath}')", language='python')
        return self.execute_code_blocks([code_block])
    
    def reset_session(self):
        """Reset the iPython shell session"""
        #self.shell.reset()
        self.shell = InteractiveShell()
    
    def execute_code_blocks(self, code_blocks, token_limit=20000):
        results = []
        for block in code_blocks:
            if block.language.lower() == 'python':
                exit_code, result = self.execute_python(block.code)
                result = result[:token_limit]
                results.append(result)
        all_results = '\n'.join(results)
        return CodeResult(exit_code=exit_code, output=all_results)

class CustomCodeExtractor(CodeExtractor):
    def extract_code_blocks(message) -> list[CodeBlock]:
        try:
            code = json.loads(message)['code']
        except json.JSONDecodeError:
            code = "# Failed to parse code from message"
        return [CodeBlock(code=code, language='python')]


def get_executor_config(shell = Optional[InteractiveShell]):
    custom_executor = CustomCodeExecutor()
    return CodeExecutionConfig(executor=custom_executor)

def get_execute_python(shell):
    def execute_python(cell: str = None):
        # Redirect stdout to capture the output
        sys.stdout = StringIO()
        try:
            # Execute the code using IPython's InteractiveShell
            result = shell.run_cell(cell)
            if result.success:
                # If execution is successful, capture the output
                output = sys.stdout.getvalue()
                log = f"Code execution: successful. Output:\n{output}"
            else:
                # If execution failed, capture the error message
                error_message = result.error_in_exec
                log = f"Code execution: failed. Output:\n{error_message}"
        except Exception as e:
            # If an exception occurs during execution, capture the error message
            log = f"Code execution: failed. Output:\n{str(e)}"
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
        
        exit_code = 0 if result.success else 1
        return exit_code, log

    return execute_python

def format_code_execution_output(
    sender: ConversableAgent,
    message: Union[dict[str, Any], str],
    recipient: Agent,
    silent: bool) -> Union[dict[str, Any], str]:

    if silent:
        return message

    output = {
        "exit_code": int(re.search(r"exitcode: (\d)", message).group(1)),
        "output": message.split('\n', 1)
    }

    return json.dumps(output)
