"""
Custom adapters for converting DSPy prompts to vanilla format.
Ensures optimized prompts are library-agnostic and compatible with any LLM client.
"""

from typing import Dict, List, Any, Optional
import json
import re


class DSPyAdapter:
    """
    Adapter to extract vanilla prompts from DSPy modules.
    
    DSPy internally injects metadata like "matches regex..." which can confuse
    other LLM clients. This adapter strips DSPy-specific formatting and produces
    clean system/user messages compatible with OpenAI, Groq, Anthropic, etc.
    """
    
    @staticmethod
    def extract_vanilla_prompt(dspy_module: Any) -> Dict[str, str]:
        """
        Extract vanilla prompt from a compiled DSPy module.
        
        Args:
            dspy_module: Compiled DSPy module (e.g., dspy.Predict, dspy.ChainOfThought)
            
        Returns:
            Dictionary with 'system' and 'user_template' keys
        """
        # Get the signature from the module
        signature = getattr(dspy_module, 'signature', None)
        if signature is None:
            raise ValueError("DSPy module has no signature")
        
        # Extract instructions (system prompt)
        instructions = getattr(dspy_module, 'extended_signature', signature).instructions
        
        # Extract few-shot examples (demonstrations)
        demos = getattr(dspy_module, 'demos', [])
        
        # Build system prompt
        system_prompt = instructions if instructions else "You are a helpful AI assistant."
        
        # Add few-shot examples to system prompt
        if demos:
            system_prompt += "\n\nHere are some examples:\n\n"
            for i, demo in enumerate(demos, 1):
                system_prompt += f"Example {i}:\n"
                # Extract input/output from demo
                for key, value in demo.items():
                    if not key.startswith('_'):  # Skip internal fields
                        system_prompt += f"{key}: {value}\n"
                system_prompt += "\n"
        
        # Build user template (with input field placeholders)
        input_fields = signature.input_fields if hasattr(signature, 'input_fields') else {}
        user_template = ""
        for field_name in input_fields:
            user_template += f"{{{field_name}}}\n"
        
        return {
            "system": system_prompt.strip(),
            "user_template": user_template.strip()
        }
    
    @staticmethod
    def clean_dspy_metadata(text: str) -> str:
        """
        Remove DSPy-specific metadata from text.
        
        Strips patterns like:
        - "matches regex ..."
        - Internal field markers
        - Type hints
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove "matches regex" patterns
        text = re.sub(r'matches regex:\s*[^\n]+', '', text)
        
        # Remove type hints like "(str)" or "(int)"
        text = re.sub(r'\(\w+\)', '', text)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def format_for_openai(system: str, user_input: str) -> List[Dict[str, str]]:
        """
        Format prompt for OpenAI-compatible API.
        
        Args:
            system: System prompt
            user_input: User message
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input}
        ]
    
    @staticmethod
    def serialize_prompt(dspy_module: Any, filepath: str) -> None:
        """
        Serialize DSPy module prompt to JSON file.
        
        Args:
            dspy_module: DSPy module to serialize
            filepath: Path to save JSON file
        """
        vanilla_prompt = DSPyAdapter.extract_vanilla_prompt(dspy_module)
        
        # Add metadata
        prompt_data = {
            "system_prompt": vanilla_prompt["system"],
            "user_template": vanilla_prompt["user_template"],
            "module_type": type(dspy_module).__name__,
            "signature": str(dspy_module.signature),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_prompt(filepath: str) -> Dict[str, str]:
        """
        Load serialized prompt from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with system and user_template
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "system": data["system_prompt"],
            "user_template": data["user_template"]
        }


class PromptFormatter:
    """
    Utility for formatting prompts with different input data.
    """
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with variable substitution.
        
        Args:
            template: Template string with {variable} placeholders
            **kwargs: Variable values to substitute
            
        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in template: {e}")
    
    @staticmethod
    def create_chat_messages(
        system_prompt: str,
        user_template: str,
        inputs: Dict[str, Any],
        include_cot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Create chat messages from prompt components.
        
        Args:
            system_prompt: System message
            user_template: User message template
            inputs: Input variables for template
            include_cot: Whether to add Chain-of-Thought instruction
            
        Returns:
            List of chat messages
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Format user message
        user_message = PromptFormatter.format_prompt(user_template, **inputs)
        
        # Add CoT instruction if requested
        if include_cot:
            user_message += "\n\nLet's think step by step."
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    @staticmethod
    def extract_output_fields(response: str, output_fields: List[str]) -> Dict[str, str]:
        """
        Extract structured output fields from LLM response.
        
        Attempts to parse response for labeled fields like "sentiment: positive".
        
        Args:
            response: LLM response text
            output_fields: Expected output field names
            
        Returns:
            Dictionary mapping field names to extracted values
        """
        result = {}
        
        for field in output_fields:
            # Try to find "field: value" pattern
            pattern = rf"{field}:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            
            if match:
                result[field] = match.group(1).strip()
            else:
                # If no explicit label, use entire response
                result[field] = response.strip()
        
        return result


class ChainOfThoughtExtractor:
    """
    Utility for extracting reasoning traces from Chain-of-Thought responses.
    """
    
    @staticmethod
    def extract_reasoning(response: str) -> Dict[str, str]:
        """
        Extract reasoning and final answer from CoT response.
        
        Args:
            response: LLM response with CoT reasoning
            
        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        # Common patterns for CoT
        answer_markers = [
            r"(?:Final Answer|Answer|Therefore):\s*(.+?)$",
            r"(?:In conclusion|Thus|So)[:,]\s*(.+?)$",
        ]
        
        answer = None
        for pattern in answer_markers:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                reasoning = response[:match.start()].strip()
                break
        
        if answer is None:
            # No explicit marker, assume last line is answer
            lines = response.strip().split('\n')
            answer = lines[-1] if lines else response
            reasoning = '\n'.join(lines[:-1]) if len(lines) > 1 else ""
        
        return {
            "reasoning": reasoning,
            "answer": answer
        }
    
    @staticmethod
    def format_cot_prompt(question: str, include_examples: bool = True) -> str:
        """
        Format a Chain-of-Thought prompt.
        
        Args:
            question: Question to answer
            include_examples: Whether to include CoT examples
            
        Returns:
            Formatted CoT prompt
        """
        prompt = ""
        
        if include_examples:
            prompt += """Here are some examples of step-by-step reasoning:

Example 1:
Question: What is 15% of 80?
Reasoning: First, I'll convert 15% to a decimal: 15/100 = 0.15
Then multiply by 80: 0.15 × 80 = 12
Answer: 12

Now answer this question:
"""
        
        prompt += f"Question: {question}\n"
        prompt += "Let's think step by step.\nReasoning:"
        
        return prompt
