from pydantic_ai import Agent, Tool
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic import Field
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pyowm import OWM 

# Load environment variables
load_dotenv()

# --- 1. Get API Keys from Environment ---
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OWM_API_KEY = os.getenv('OWM_API_KEY')

if not TAVILY_API_KEY:
    print("Error: TAVILY_API_KEY not found in environment variables.")
    exit()
if not OWM_API_KEY:
    print("Error: OWM_API_KEY not found in environment variables.")
    exit()

# --- 2. Define the Custom OWM Tool (Final Working Version) ---
class OpenWeatherMapTool(Tool):
    """A tool to get the current weather conditions for a specific location using the OpenWeatherMap API."""
    
    # Field name 'city' matches LLM's preferred output
    city: str = Field(description="The city (e.g., 'Bangalore' or 'London,GB') for which to get the current weather.")

    # FIX: Parameter 'city' is explicitly passed and used.
    def run(self, city: str) -> str:
        """The main function executed when the LLM decides to use this tool."""
        try:
            # Initialize OWM client here
            owm = OWM(OWM_API_KEY)
            mgr = owm.weather_manager()
            
            # Use the passed 'city' argument directly
            observation = mgr.weather_at_place(city)
            w = observation.weather
            
            # Format the data into a useful string
            temp_c = w.temperature('celsius')['temp']
            status = w.detailed_status
            wind = w.wind()
            
            return (
                f"Current weather in {city}: "
                f"Temperature is {temp_c:.1f}°C. "
                f"Conditions are '{status}'. "
                f"Wind speed is {wind['speed']:.1f} m/s."
            )
        except Exception as e:
            # Handle API/location errors gracefully, using 'city'
            return f"Error: Could not retrieve weather for {city}. Details: {e}"


# --- 3. Initialize Agent with Both Tools and Llama-3.3 ---
console = Console()
agent = Agent(
    'groq:llama-3.3-70b-versatile', 
    
    tools=[
        tavily_search_tool(TAVILY_API_KEY), 
        # Pass the tool instance, specifying the run method as the callable function
        OpenWeatherMapTool(function=OpenWeatherMapTool.run) 
    ], 
    instructions=(
        'You are a helpful assistant named OverSight. '
        'For **ANY** query related to current weather, temperature, or conditions, you **MUST** use the `OpenWeatherMapTool`. '
        'Do not use the search tool for weather. '
        'For all other factual questions, use the search tool. '
        'Keep your answers brief and concise.'
    ),
)

# --- 4. Start the Chat Loop ---
console.print(Panel(
    Text("Welcome to OverSight! I can search the web and check the weather. Type 'exit' to quit.", style="bold yellow"),
    title="Agent Chat",
    title_align="center"
))

while True:
    user_prompt = console.input("[bold green]You: [/]")
    
    if user_prompt.lower() == 'exit':
        console.print(Panel("Goodbye!", title="OverSight", style="bold red"))
        break

    try:
        console.print(Panel(
            Text("Thinking...", style="italic dim"), 
            border_style="dim"
        ))

        # Run the agent with the user's prompt
        result = agent.run_sync(user_prompt)
        
        # Display the agent's output
        agent_output = result.output
        console.print(Panel(
            Text(f"{agent_output}", style="cyan"), 
            title="OverSight", 
            border_style="cyan"
        ))
    except Exception as e:
        console.print(Panel(
            Text(f"An error occurred: {e}", style="bold red"), 
            title="Error",
            border_style="red"
        ))