import streamlit as st
import requests
import time
import os
import json
import pandas as pd
from dataset_analyzer import DatasetAnalyzer

# Load environment variables from .env file if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    # dotenv not installed, likely running on Replit with secrets
    print("python-dotenv not installed, using environment variables directly")

def display_ai_assistant():
    st.title("🤖 AI Assistant")
    st.write("Chat with our AI assistant powered by Llama 4 to get help with your biomedical data analysis")
    
    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "api_call_counter" not in st.session_state:
        st.session_state.api_call_counter = 0
        
    if "rate_limit_reached" not in st.session_state:
        st.session_state.rate_limit_reached = False
    
    if "service_provider" not in st.session_state:
        st.session_state.service_provider = "OpenRouter.ai (Llama 4 Maverick)"
    
    # Initialize dataset analyzer if it doesn't exist
    if "dataset_analyzer" not in st.session_state:
        st.session_state.dataset_analyzer = DatasetAnalyzer()
        
    # Update dataset analyzer with the current dataset if one is loaded
    if st.session_state.get("raw_df") is not None:
        dataset_df = st.session_state.cleaned_df if st.session_state.get("cleaned_df") is not None else st.session_state.raw_df
        st.session_state.dataset_analyzer.set_dataframe(dataset_df)
    
    # Display dataset analysis features
    with st.sidebar.expander("🔍 Dataset Analysis Options", expanded=False):
        if st.session_state.get("raw_df") is not None:
            st.write("Dataset loaded and available for analysis")
            
            # Add analysis options
            if st.button("Analyze Current Dataset"):
                with st.spinner("Analyzing dataset..."):
                    analysis_results = st.session_state.dataset_analyzer.analyze_dataset()
                    st.session_state.analysis_summary = st.session_state.dataset_analyzer.generate_summary()
                    st.success("Analysis complete!")
                    
                    # Add analysis results to chat as assistant message
                    if "analysis_summary" in st.session_state:
                        analysis_msg = (
                            "📊 **Dataset Analysis Complete**\n\n" + 
                            st.session_state.analysis_summary + 
                            "\n\nYou can now ask specific questions about the dataset, dimensionality reduction, "
                            "visualization recommendations, or machine learning model selection."
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": analysis_msg})
                        st.rerun()
        else:
            st.write("Please upload a dataset first using the sidebar uploader")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    user_input = st.chat_input("Ask me about your biomedical data...")
    
    # Handle user input
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Create thinking message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write("Thinking...")
            
            # Enhance the user input with dataset context if dataset is loaded
            enhanced_input = user_input
            dataset_context = ""
            
            if st.session_state.get("raw_df") is not None:
                # Check if we need to analyze the dataset first
                if not hasattr(st.session_state.dataset_analyzer, 'analysis_results') or not st.session_state.dataset_analyzer.analysis_results:
                    with st.spinner("Analyzing dataset for context..."):
                        st.session_state.dataset_analyzer.analyze_dataset()
                
                # Check if user is asking about dataset, visualizations, dimensions, or ML
                query_terms = [
                    "dataset", "data", "column", "feature", "missing", "values",
                    "visualization", "chart", "plot", "graph", "dimensionality", 
                    "pca", "tsne", "umap", "lda", "reduction", "machine learning",
                    "model", "classification", "regression", "predict"
                ]
                
                if any(term in user_input.lower() for term in query_terms):
                    try:
                        # Get dataset summary and recommendations relevant to the query
                        if "dimension" in user_input.lower() or "pca" in user_input.lower() or "tsne" in user_input.lower() or "umap" in user_input.lower():
                            dim_recommendations = st.session_state.dataset_analyzer.get_dimensionality_reduction_recommendations()
                            dim_json = json.dumps(dim_recommendations, default=str)  # Use default=str as fallback
                            dataset_context = f"The dataset has these dimensionality reduction recommendations: {dim_json}\n"
                        
                        elif "model" in user_input.lower() or "machine learning" in user_input.lower() or "classification" in user_input.lower() or "regression" in user_input.lower():
                            ml_recommendations = st.session_state.dataset_analyzer.get_ml_model_recommendations()
                            ml_json = json.dumps(ml_recommendations, default=str)  # Use default=str as fallback
                            dataset_context = f"The dataset has these machine learning model recommendations: {ml_json}\n"
                        
                        elif "visual" in user_input.lower() or "chart" in user_input.lower() or "plot" in user_input.lower() or "graph" in user_input.lower():
                            viz_recommendations = st.session_state.dataset_analyzer.get_visualization_recommendations()
                            viz_json = json.dumps(viz_recommendations, default=str)  # Use default=str as fallback
                            dataset_context = f"The dataset has these visualization recommendations: {viz_json}\n"
                        
                        else:
                            # General dataset information
                            dataset_context = st.session_state.dataset_analyzer.get_dataset_json_for_llm()
                            
                        # Enhance the user input with context
                        enhanced_input = f"Context about the current dataset: {dataset_context}\n\nUser question: {user_input}"
                    
                    except Exception as e:
                        # If there's any error during context creation, continue without context
                        st.error(f"Error preparing dataset context: {str(e)}")
                        enhanced_input = user_input
                        # Add general dataset info as a fallback
                        try:
                            rows = len(st.session_state.raw_df)
                            cols = len(st.session_state.raw_df.columns)
                            enhanced_input = f"This dataset has {rows} rows and {cols} columns.\n\nUser question: {user_input}"
                        except:
                            pass
            
            # Get AI response
            try:
                # Check which API to use
                if st.session_state.rate_limit_reached:
                    # Use Meta Llama 3 as fallback
                    response_content = get_meta_llama_response(enhanced_input)
                    st.session_state.service_provider = "Meta Llama 3 (Fallback)"
                else:
                    # Try OpenRouter.ai Llama 4 Maverick first
                    try:
                        response_content = get_openrouter_response(enhanced_input)
                        st.session_state.api_call_counter += 1
                        st.session_state.service_provider = "OpenRouter.ai (Llama 4 Maverick)"
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            st.session_state.rate_limit_reached = True
                            response_content = get_meta_llama_response(enhanced_input)
                            st.session_state.service_provider = "Meta Llama 3 (Fallback)"
                        else:
                            raise e
                
                # Update the message
                message_placeholder.write(response_content)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                
            except Exception as e:
                message_placeholder.write(f"Error: {str(e)}")
                st.error(f"Failed to get response: {str(e)}")
    
    # Show which service is being used
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current AI Provider:** {st.session_state.service_provider}")
    
    # Add reset button to sidebar
    if st.sidebar.button("Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.rate_limit_reached = False
        st.session_state.service_provider = "OpenRouter.ai (Llama 4 Maverick)"
        st.rerun()

# Hardcoded API Keys (Use with caution)
OPENROUTER_API_KEY = "" ### Add your API
REPLICATE_API_TOKEN = "" ### Add your API


def get_openrouter_response(user_input):
    """
    Get response from OpenRouter.ai using Llama 4 Maverick
    """
    # Get API key from environment variable with fallback
    api_key = OPENROUTER_API_KEY


    
    if not api_key:
        return """
        **API Key Missing**
        
        To use the AI Assistant, you need to set up an OpenRouter.ai API key:
        
        1. Sign up at [OpenRouter.ai](https://openrouter.ai/)
        2. Navigate to your account page and create a new API key
        3. Set the key as an environment variable:
           
           **On Replit:**
           - Add the key as a Secret named `OPENROUTER_API_KEY` in the Replit Secrets panel
           
           **Running Locally:**
           - Create a `.env` file in the root directory with:
             ```
             OPENROUTER_API_KEY=your_key_here
             ```
             
        Restart the application after setting up your API key.
        """
    
    # Format messages including chat history
    messages = []
    for message in st.session_state.chat_history[-10:]:  # Include last 10 messages for context
        messages.append({"role": message["role"], "content": message["content"]})
    
    # Add the current user message
    if not any(msg["role"] == "user" and msg["content"] == user_input for msg in messages):
        messages.append({"role": "user", "content": user_input})
    
    # Define system message
    system_message = """You are an AI assistant specialized in biomedical data analysis with advanced dataset understanding capabilities.
    
    Your capabilities include:
    1. Analyzing datasets to provide statistics, data distributions, and patterns
    2. Recommending appropriate visualizations based on data types
    3. Suggesting optimal dimensionality reduction techniques (PCA, t-SNE, UMAP, LDA) with parameter recommendations
    4. Recommending machine learning models (classification/regression) suited to the dataset
    5. Providing data cleaning and preprocessing advice
    
    When users ask about their dataset, provide specific recommendations with explanations. Use the dataset context
    provided to give personalized advice rather than generic responses. If users ask about machine learning models, 
    dimensionality reduction, or visualizations, explain why your recommendations are appropriate for their specific dataset.
    
    Be precise, informative, and helpful. Explain concepts in clear terms appropriate for biomedical researchers."""
    
    # Prepare request data - Using OpenRouter.ai format from their documentation
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",  # OpenRouter requires "Bearer" prefix
        "HTTP-Referer": "https://replit.com",  # Helps with API usage tracking
        "X-Title": "Biomedical Data Assistant"  # Helps with API usage tracking
    }
    
    payload = {
        "model": "meta-llama/llama-4-maverick:free",  # Using Llama 4 Maverick free tier
        "messages": [
            {"role": "system", "content": system_message},
            *messages
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    # Make the request
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        # Display detailed debug info for troubleshooting
        if response.status_code != 200:
            error_detail = f"Status Code: {response.status_code}\n"
            try:
                error_detail += f"Response: {response.json()}\n"
            except:
                error_detail += f"Raw Response: {response.text}\n"
            
            # Log headers for debugging (without the actual API key)
            safe_headers = headers.copy()
            if "Authorization" in safe_headers:
                safe_headers["Authorization"] = "Bearer sk-or-***" # Redacted for security
            error_detail += f"Headers: {safe_headers}\n"
            
            st.error(error_detail)
        
        # Check for errors
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.session_state.rate_limit_reached = True
            raise Exception("Rate limit reached for OpenRouter.ai. Switching to fallback model.")
        elif e.response.status_code == 401:
            # More descriptive error for auth issues
            raise Exception("Authentication failed. Please check your OpenRouter API key - it should start with 'sk-or-'")
        else:
            raise Exception(f"HTTP error occurred: {str(e)}")
    except Exception as e:
        raise Exception(f"Error getting response from OpenRouter.ai: {str(e)}")


def get_meta_llama_response(user_input):
    """
    Get response from Replicate's Llama 3 API (fallback)
    """
    # Get API key from environment variable with fallback
    api_key = REPLICATE_API_TOKEN


    
    if not api_key:
        return """
        **Fallback API Key Missing**
        
        The primary AI service (OpenRouter.ai) is unavailable or has reached its rate limit. 
        To use the fallback AI Assistant, you need to set up a Replicate API token:
        
        1. Create an account at [Replicate](https://replicate.com/)
        2. Go to your account settings and find or create an API token
        3. Set the token as an environment variable:
           
           **On Replit:**
           - Add the token as a Secret named `REPLICATE_API_TOKEN` in the Replit Secrets panel
           
           **Running Locally:**
           - Create a `.env` file in the root directory with:
             ```
             REPLICATE_API_TOKEN=your_token_here
             ```
             
        Restart the application after setting up your API token.
        """
    
    # Format messages including chat history
    conversation = []
    for message in st.session_state.chat_history[-10:]:  # Include last 10 messages for context
        conversation.append({"role": message["role"], "content": message["content"]})
    
    # Add the current user message
    if not any(msg["role"] == "user" and msg["content"] == user_input for msg in conversation):
        conversation.append({"role": "user", "content": user_input})
    
    # Define system message
    system_message = """You are an AI assistant specialized in biomedical data analysis with advanced dataset understanding capabilities.
    
    Your capabilities include:
    1. Analyzing datasets to provide statistics, data distributions, and patterns
    2. Recommending appropriate visualizations based on data types
    3. Suggesting optimal dimensionality reduction techniques (PCA, t-SNE, UMAP, LDA) with parameter recommendations
    4. Recommending machine learning models (classification/regression) suited to the dataset
    5. Providing data cleaning and preprocessing advice
    
    When users ask about their dataset, provide specific recommendations with explanations. Use the dataset context
    provided to give personalized advice rather than generic responses. If users ask about machine learning models, 
    dimensionality reduction, or visualizations, explain why your recommendations are appropriate for their specific dataset.
    
    Be precise, informative, and helpful. Explain concepts in clear terms appropriate for biomedical researchers."""
    
    # Prepare request data for Replicate API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}"
    }
    
    # Format the input for Replicate API
    messages = [{"role": "system", "content": system_message}]
    messages.extend(conversation)
    
    # Create a prediction
    payload = {
        "version": "2d19859030ff705a87c746f7e96eea03aefb71f166725fac8c5a287d58000d8f",  # Llama 3 model ID
        "input": {
            "prompt": "",  # We'll use the messages format instead
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "system_prompt": system_message
        }
    }
    
    # Make the request
    try:
        # Create the prediction
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Get the prediction data
        prediction = response.json()
        prediction_id = prediction["id"]
        
        # Poll for the prediction result
        while True:
            time.sleep(1)
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )
            response.raise_for_status()
            prediction = response.json()
            
            if prediction["status"] == "succeeded":
                # Extract the response - Replicate returns an array of strings
                return "".join(prediction["output"])
            elif prediction["status"] == "failed":
                return f"Error: Prediction failed. Reason: {prediction.get('error', 'Unknown error')}"
            
            # Wait a bit before polling again
            time.sleep(1)
    
    except Exception as e:
        return f"Error getting response from Replicate API: {str(e)}\n\nPlease try again later or contact support for assistance."

def simulate_ai_response(user_input):
    """
    Simulate an AI response for testing when APIs are not available
    This is only used for development and should be removed in production
    """
    time.sleep(1)  # Simulate thinking time
    
    responses = {
        "hello": "Hello! I'm your biomedical data analysis assistant. How can I help you today?",
        "help": "I can help you with various biomedical data tasks like interpreting results, suggesting analysis approaches, explaining medical concepts, and more. Just ask me a specific question about your data or analysis.",
        "default": "I'm here to help with your biomedical data analysis questions. Could you provide more details about what you'd like assistance with?"
    }
    
    # Look for keywords in user input to determine response
    user_input_lower = user_input.lower()
    
    if "hello" in user_input_lower or "hi" in user_input_lower:
        return responses["hello"]
    elif "help" in user_input_lower:
        return responses["help"]
    else:
        return responses["default"] + "\n\nFor example, you could ask about:\n- Statistical methods for your biomedical data\n- Interpreting visualization results\n- Understanding medical imaging analysis\n- Best practices for data cleaning in biomedical research"
