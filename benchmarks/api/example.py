"""
LLM API Usage Examples
Demonstrates various calling methods and use cases
"""

# ============================================================================
# Example 1: Using Configuration File (Simplest)
# ============================================================================
def example1_use_config():
    """Load and use from configuration file"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 1: Using Configuration File")
    print("=" * 60)

    # Create client from configuration file
    client = get_client_from_config("gemini")

    # Generate text
    response = client.generate("Explain what AI is in one sentence")
    print(f"Answer: {response}\n")


# ============================================================================
# Example 2: Direct Parameters
# ============================================================================
def example2_direct_params():
    """Pass configuration parameters directly"""
    from api import get_client

    print("=" * 60)
    print("Example 2: Direct Parameters")
    print("=" * 60)

    # Gemini
    gemini_client = get_client(
        "gemini",
        project="your-project",
        location="us-central1",
        model_name="gemini-2.5-pro",
        credentials_path="path/to/credentials.json"
    )

    # DeepSeek
    deepseek_client = get_client(
        "deepseek",
        api_key="your-api-key",
        appid="your-appid",
        base_url="https://qianfan.baidubce.com/v2"
    )

    # Usage
    response = gemini_client.generate("Hello")
    print(f"Gemini: {response}\n")


# ============================================================================
# Example 3: Batch Generation (Concurrent)
# ============================================================================
def example3_batch_generate():
    """Batch text generation with concurrent support"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 3: Batch Generation (Concurrent)")
    print("=" * 60)

    prompts = [
        "What is machine learning?",
        "Explain deep learning",
        "Principles of neural networks",
        "What is natural language processing?",
        "Applications of computer vision"
    ]

    # Use client instance's batch_generate method (recommended)
    client = get_client_from_config("gemini")
    results = client.batch_generate(
        prompts=prompts,
        max_workers=3,  # 3 concurrent threads
        show_progress=True  # Show progress bar
    )

    # Process results
    for i, item in enumerate(results, 1):
        print(f"\nQuestion {i}: {item['prompt']}")
        if item['success']:
            print(f"Answer: {item['result'][:100]}...")
        else:
            print(f"Error: {item['error']}")


# ============================================================================
# Example 4: Custom Generation Parameters
# ============================================================================
def example4_custom_params():
    """Custom generation parameters"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 4: Custom Generation Parameters")
    print("=" * 60)

    client = get_client_from_config("deepseek")

    # Creative generation (high temperature)
    creative = client.generate(
        "Write a poem about spring",
        temperature=0.9,
        max_tokens=200
    )
    print(f"Creative output:\n{creative}\n")

    # Precise generation (low temperature)
    precise = client.generate(
        "What is 1+1?",
        temperature=0.1,
        max_tokens=50
    )
    print(f"Precise output:\n{precise}\n")


# ============================================================================
# Example 5: Error Handling
# ============================================================================
def example5_error_handling():
    """Demonstrate error handling"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    try:
        client = get_client_from_config("gemini")

        # Normal call
        response = client.generate("Hello")
        print(f"Success: {response}")

        # Empty prompt (will raise ValueError)
        response = client.generate("")

    except ValueError as e:
        print(f"Parameter error: {e}")
    except Exception as e:
        print(f"API call failed: {e}")


# ============================================================================
# Example 6: Switch Models
# ============================================================================
def example6_switch_models():
    """Switch between different models"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 6: Switch Models")
    print("=" * 60)

    question = "What is quantum computing?"

    for model_name in ["gemini", "deepseek"]:
        try:
            client = get_client_from_config(model_name)
            response = client.generate(question)
            print(f"\n{model_name.upper()}'s answer:")
            print(response[:150] + "...")
        except Exception as e:
            print(f"\n{model_name} call failed: {e}")


# ============================================================================
# Example 7: Real Application - User Profile Generation
# ============================================================================
def example7_user_portrait():
    """Real application: Generate user profile based on user behavior"""
    from api import get_client_from_config

    print("=" * 60)
    print("Example 7: User Profile Generation")
    print("=" * 60)

    # User behavior data
    user_behavior = """
    User's recently watched videos:
    1. Machine Learning Tutorial
    2. Python Programming Tips
    3. Deep Learning Practical Projects
    4. Data Analysis Case Studies
    5. Latest AI Trends
    """

    prompt = f"""Based on the following user behavior data, generate a concise user profile:

{user_behavior}

Requirements:
1. Summarize user's areas of interest
2. Infer user's skill level
3. Provide 3-5 precise tags
"""

    client = get_client_from_config("gemini")
    portrait = client.generate(prompt, temperature=0.5)

    print("User Profile:")
    print(portrait)


# ============================================================================
# Example 8: Direct Import of Classes
# ============================================================================
def example8_direct_import():
    """Import client classes directly"""
    from api import GeminiClient, DeepSeekClient

    print("=" * 60)
    print("Example 8: Direct Import of Client Classes")
    print("=" * 60)

    # Direct instantiation
    gemini = GeminiClient(
        project="your-project",
        location="us-central1"
    )

    deepseek = DeepSeekClient(
        api_key="your-key",
        appid="your-appid"
    )

    print("Clients created successfully")
    print(f"Gemini client: {gemini}")
    print(f"DeepSeek client: {deepseek}")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Run all examples"""
    examples = [
        ("Using Configuration File", example1_use_config),
        ("Direct Parameters", example2_direct_params),
        ("Batch Generation", example3_batch_generate),
        ("Custom Parameters", example4_custom_params),
        ("Error Handling", example5_error_handling),
        ("Switch Models", example6_switch_models),
        ("User Profile Generation", example7_user_portrait),
        ("Direct Import Classes", example8_direct_import),
    ]

    print("\n" + "=" * 60)
    print("LLM API Usage Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nNote: Please ensure api/config/llm_config.json is configured before running")
    print("\n" + "=" * 60 + "\n")

    # Uncomment the lines below to run specific examples
    # example1_use_config()
    # example2_direct_params()
    # example3_batch_generate()
    # example4_custom_params()
    # example5_error_handling()
    # example6_switch_models()
    # example7_user_portrait()
    # example8_direct_import()


if __name__ == "__main__":
    main()
