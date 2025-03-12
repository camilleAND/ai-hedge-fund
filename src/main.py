import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes, ANALYST_CONFIG
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info, AVAILABLE_MODELS

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None



##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


def load_config_file(config_path):
    """Load and validate configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['tickers', 'analysts', 'model']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")
        
        # Validate tickers
        if not isinstance(config['tickers'], list) or not config['tickers']:
            raise ValueError("'tickers' must be a non-empty list of strings")
        
        # Validate analysts
        if not isinstance(config['analysts'], list) or not config['analysts']:
            raise ValueError("'analysts' must be a non-empty list of strings")
        
        # Check if analysts are valid
        valid_analysts = set(key for key in ANALYST_CONFIG.keys())
        invalid_analysts = [analyst for analyst in config['analysts'] if analyst not in valid_analysts]
        if invalid_analysts:
            raise ValueError(f"Invalid analysts in config file: {', '.join(invalid_analysts)}. Valid options are: {', '.join(valid_analysts)}")
        
        # Validate model
        if not isinstance(config['model'], str) or not config['model']:
            raise ValueError("'model' must be a non-empty string")
        
        # Check if model is valid
        valid_models = [model.model_name for model in AVAILABLE_MODELS]
        if config['model'] not in valid_models:
            raise ValueError(f"Invalid model in config file: {config['model']}. Valid options are: {', '.join(valid_models)}")
        
        return config
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0)"
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0"
    )
    parser.add_argument(
        "--tickers", 
        type=str, 
        required=not any('--script' in arg for arg in sys.argv),
        help="Comma-separated list of stock ticker symbols (not required when using --script)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument(
        "--show-agent-graph", action="store_true", help="Show the agent graph"
    )
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    parser.add_argument("--script", action="store_true", help="Run in script mode using configuration from a JSON file")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file (required when using --script)")
    parser.add_argument(
        "--analysts",
        type=str,
        help="Comma-separated list of analysts to use (bypasses interactive selection)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (bypasses interactive selection)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode with command line arguments"
    )
    
    args = parser.parse_args()

    # Check if script mode requires a config file
    if args.script and not args.config:
        parser.error("--script mode requires a --config JSON file")
    
    # Check if non-interactive mode has required arguments
    if args.non_interactive and not all([args.tickers, args.analysts, args.model]):
        parser.error("Non-interactive mode requires --tickers, --analysts, and --model arguments")
    
    # Load configuration from JSON file if in script mode
    if args.script:
        try:
            config = load_config_file(args.config)
            tickers = config['tickers']
            selected_analysts = config['analysts']
            model_choice = config['model']
            
            # Optional configuration parameters
            start_date = config.get('start_date', args.start_date)
            end_date = config.get('end_date', args.end_date)
            show_reasoning = config.get('show_reasoning', args.show_reasoning)
            show_agent_graph = config.get('show_agent_graph', args.show_agent_graph)
            initial_cash = config.get('initial_cash', args.initial_cash)
            margin_requirement = config.get('margin_requirement', args.margin_requirement)
            
            # Get model info
            model_info = get_model_info(model_choice)
            if model_info:
                model_provider = model_info.provider.value
                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
            else:
                model_provider = "Unknown"
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
                
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in selected_analysts)}\n")
            print(f"\nAnalyzing tickers: {', '.join(Fore.YELLOW + ticker + Style.RESET_ALL for ticker in tickers)}\n")
            
        except (ValueError, FileNotFoundError) as e:
            print(f"{Fore.RED}Error loading configuration: {e}{Style.RESET_ALL}")
            sys.exit(1)
    # Non-interactive mode with command line arguments
    elif args.non_interactive:
        # Parse tickers from comma-separated string
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
        print(f"\nAnalyzing tickers: {', '.join(Fore.YELLOW + ticker + Style.RESET_ALL for ticker in tickers)}\n")
        
        # Parse analysts from comma-separated string
        selected_analysts = [analyst.strip() for analyst in args.analysts.split(",")]
        
        # Validate analysts
        valid_analysts = set(key for key in ANALYST_CONFIG.keys())
        invalid_analysts = [analyst for analyst in selected_analysts if analyst not in valid_analysts]
        if invalid_analysts:
            print(f"{Fore.RED}Error: Invalid analysts: {', '.join(invalid_analysts)}{Style.RESET_ALL}")
            print(f"Valid options are: {', '.join(valid_analysts)}")
            sys.exit(1)
            
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in selected_analysts)}\n")
        
        # Validate model
        model_choice = args.model
        valid_models = [model.model_name for model in AVAILABLE_MODELS]
        if model_choice not in valid_models:
            print(f"{Fore.RED}Error: Invalid model: {model_choice}{Style.RESET_ALL}")
            print(f"Valid options are: {', '.join(valid_models)}")
            sys.exit(1)
            
        # Get model info
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    else:
        # Parse tickers from comma-separated string
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
        print(f"\nAnalyzing tickers: {', '.join(Fore.YELLOW + ticker + Style.RESET_ALL for ticker in tickers)}\n")

        # Select analysts
        selected_analysts = None
        choices = questionary.checkbox(
            "Select your AI analysts.",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
            validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
            style=questionary.Style(
                [
                    ("checkbox-selected", "fg:green"),
                    ("selected", "fg:green noinherit"),
                    ("highlighted", "noinherit"),
                    ("pointer", "noinherit"),
                ]
            ),
        ).ask()

        if not choices:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            selected_analysts = choices
            print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

        # Select LLM model
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
            style=questionary.Style([
                ("selected", "fg:green bold"),
                ("pointer", "fg:green bold"),
                ("highlighted", "fg:green"),
                ("answer", "fg:green bold"),
            ])
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)
        else:
            # Get model info using the helper function
            model_info = get_model_info(model_choice)
            if model_info:
                model_provider = model_info.provider.value
                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
            else:
                model_provider = "Unknown"
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.script:
        # Use dates from config if available
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Start date in config must be in YYYY-MM-DD format")
        
        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("End date in config must be in YYYY-MM-DD format")
    else:
        # Validate dates if provided through command line
        if args.start_date:
            try:
                datetime.strptime(args.start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Start date must be in YYYY-MM-DD format")

        if args.end_date:
            try:
                datetime.strptime(args.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    if args.script:
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Calculate 3 months before end_date
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        if not args.start_date:
            # Calculate 3 months before end_date
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
        else:
            start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    if args.script:
        portfolio = {
            "cash": initial_cash,  # Initial cash amount
            "margin_requirement": margin_requirement,  # Initial margin requirement
            "positions": {
                ticker: {
                    "long": 0,  # Number of shares held long
                    "short": 0,  # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis for long positions
                    "short_cost_basis": 0.0,  # Average price at which shares were sold short
                } for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                } for ticker in tickers
            }
        }
    else:
        portfolio = {
            "cash": args.initial_cash,  # Initial cash amount
            "margin_requirement": args.margin_requirement,  # Initial margin requirement
            "positions": {
                ticker: {
                    "long": 0,  # Number of shares held long
                    "short": 0,  # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis for long positions
                    "short_cost_basis": 0.0,  # Average price at which shares were sold short
                } for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                } for ticker in tickers
            }
        }

    # Run the hedge fund
    if args.script:
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_choice,
            model_provider=model_provider,
        )
    else:
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=args.show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_choice,
            model_provider=model_provider,
        )
    print_trading_output(result)
