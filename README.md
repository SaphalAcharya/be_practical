# Gemini Agentic AI - Custom Tool Calling

## Project Overview

This project demonstrates how to build an agentic AI application using the Google Gemini API and Python. The agent can use custom Python functions as external tools to retrieve information and provide a final response to the user.

## Objective

The objective of this project is to understand:

* Gemini API integration
* Function/tool calling
* Custom Python tools
* Type hints and docstrings
* Agent-based task execution
* Passing tool results back to the AI model

## Technologies Used

* Python
* Google Gemini API
* Google GenAI SDK
* Google Colab / VS Code
* Pandas
* Matplotlib
* Seaborn

## Custom Tool

The project includes a custom flight-status tool:

```python
check_flight_status(flight_number: str)
```

The function uses a mock database to return information such as:

* Flight status
* Departure location
* Arrival location
* Departure time

## Example

The agent receives the following request:

```text
What is the current status of flight DL404?
```

The Gemini model identifies that the flight-status tool is required, calls the Python function, receives the result, and generates the final response.

## Agent Workflow

```text
User Query
    ↓
Gemini Model
    ↓
Tool Selection
    ↓
check_flight_status()
    ↓
Mock Flight Database
    ↓
Tool Result
    ↓
Gemini Model
    ↓
Final Response
```

## Installation

Install the required Google GenAI SDK:

```bash
pip install -U google-genai
```

## API Key Setup

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not upload your API key to GitHub.

Add `.env` to `.gitignore`:

```text
.env
```

## Running the Project

Run the Python notebook or script after configuring the Gemini API key.

The agent will process the user request and use the custom flight-status function when required.

## Sample Output

```text
=== FLIGHT STATUS AGENT ===

User: What is the current status of flight DL404?

Agent Response:
Flight DL404 is On Time.
Departure: Delhi.
Arrival: New York.
Departure time: 10:30 AM.
```

## Learning Outcome

This project demonstrates how external Python functions can extend the capabilities of a large language model. Instead of generating an answer only from its internal knowledge, the Gemini agent can call an external tool, obtain information, and use that information to generate its final response.

## Author

Saphal Acharya
