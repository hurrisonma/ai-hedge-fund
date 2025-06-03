"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory=None,
    debug: bool = False,
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        debug: Whether to print debug information about the API request

    Returns:
        An instance of the specified Pydantic model
    """
    from src.llm.models import ModelProvider
    
    model_info = get_model_info(model_name, model_provider)
    # Convert string to ModelProvider enum
    provider_enum = ModelProvider(model_provider)
    llm = get_model(model_name, provider_enum)

    # Debug: Print API request information
    if debug:
        print(f"\n{'='*60}")
        print(f"🔍 API REQUEST DEBUG INFO")
        print(f"{'='*60}")
        print(f"📍 Agent: {agent_name or 'Unknown'}")
        print(f"🤖 Model: {model_name}")
        print(f"🏢 Provider: {model_provider}")
        print(f"📝 Pydantic Model: {pydantic_model.__name__}")
        print(f"🔧 Has JSON Mode: {model_info.has_json_mode() if model_info else 'Unknown'}")
        print(f"\n📨 PROMPT CONTENT:")
        print(f"{'-'*40}")
        if hasattr(prompt, 'messages'):
            for i, message in enumerate(prompt.messages):
                if hasattr(message, 'content'):
                    print(f"Message {i+1} ({type(message).__name__}):")
                    print(f"{message.content}")
                    print(f"{'-'*40}")
        else:
            print(f"{prompt}")
        print(f"{'='*60}\n")

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # Debug: Print API response information
            if debug:
                print(f"✅ API RESPONSE DEBUG INFO")
                print(f"{'-'*40}")
                print(f"🔄 Attempt: {attempt + 1}")
                print(f"📤 Response Type: {type(result).__name__}")
                if hasattr(result, 'content'):
                    print(f"📝 Response Content (first 500 chars):")
                    print(f"{str(result.content)[:500]}...")
                else:
                    print(f"📝 Response: {str(result)[:500]}...")
                print(f"{'-'*40}\n")

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if debug:
                print(f"❌ API ERROR DEBUG INFO")
                print(f"{'-'*40}")
                print(f"🔄 Attempt: {attempt + 1}/{max_retries}")
                print(f"❌ Error: {str(e)}")
                print(f"{'-'*40}\n")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None
