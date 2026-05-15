import autogen
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen import register_function

from pandasai import Agent, SmartDataframe
from pandasai.skills import skill
from pandasai.llm import AzureOpenAI

from huggingface_hub import login

import os
import glob
import json
import csv
import numpy as np
import pandas as pd
import cv2
from typing import Annotated, List, Optional, Union

from functions.create_hf_dataset import prepare_dataset, get_dataset_format
from functions.instance_segmentation import finetune_instance_segmentation, infer_instance_segmentation
from functions.image_classification import finetune_image_classification, infer_image_classification, infer_image_classification_dinov2
from functions.image_regression import finetune_image_regression, infer_image_regression
from functions.search import search_and_scrape
from functions.multimodal_models import (
    infer_yolo_object_detection,
    infer_temporal_crop_model,
)
from functions.compute_phenotypes import compute_phenotypes_from_ins_seg
from functions.reproducible_pipeline import save_pipeline, load_chat_log, get_pipeline_zoo, get_pipeline_info, execute_pipeline
from functions.stat_test import perform_anova, perform_tukey_test
from functions.generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images,  
    get_model_zoo,
    calculator,
    make_dir,
    extract_column_name_from_csv,
)

set_env_vars("./.env.yaml")
login(token=os.environ['HF_TOKEN'])
WORK_DIR = "./"
MODEL_ZOO_PATH = "./model_zoo.json"

# LLM configs
config_list = [
  {
    "model": os.environ['MODEL_NAME'],
    # "model": "gpt-4o",
    "base_url": os.environ['AZURE_API_URL'],
    "api_version": os.environ['AZURE_API_VERSION'],
    "temperature": 0.1,
    "cache_seed": 42,
    "timeout": 540000,
    "api_type": "azure",
    "api_key": os.environ['OPENAI_API_KEY'],
  }
]

gpt_config = {
    "config_list": config_list,
}

MANAGER_SYSTEM_PROMPT = '''
manager. Use the available tools to solve plant-related tasks.

Instructions:
1. Begin by creating a clear, step-by-step plan for the task. Provide an explanation of each step before proceeding. You may refine the plan as needed based on intermediate results.
2. When using multiple tools, be sure to provide the outputs from one tool to the next as inputs if necessary.
3. When you need to use coding-related tools, provide clear descriptions of the task requirements and expected outputs. Do not pass concrete codes to the coding related functions.
4. When you need to use computer vision to solve a task, first check the available checkpoints using the 'get_model_zoo' function:
    - Select a suitable checkpoint for the task if it it exists. Make sure you pass a list of image paths (e.g. [./data/image1.png, ./data/image2.png, ...]), or a single csv/json file (e.g. ./data/metadata.json) to the corresponding inference function.
    - If no suitable checkpoint is available, suggest user to finetune a new model. DO NOT rely on online source. Use the 'get_dataset_format' function to instruct the user to provide a training dataset.
5. When a user ask you to execute a saved pipeline, first call 'get_pipeline_zoo' to know what pipelines are available, and then call 'get_pipeline_info' to understand how to use the selected pipeline.
6. At the end of the task, provide a summary of the results and ask the user if they need any further assistance.
    
Return "TERMINATE" when the task is completed.
'''

PIPELINE_SUMMARISER_SYSTEM_PROMPT = """pipeline_summariser. Extract executed function calls and Python code snippets from a chat log, in order to form a pipeline that can be reused in the future. 
    Instructions:
      - Output the pipeline as a single Python function with key arguments.
      - Provide a clear description, annotate all key arguments with their data types and descriptions, and specify the output type.
      - Include only the executed function calls and Python code snippets. Do not include the unexcuted ones.
      - For code snippets, include them exactly as they were executed, including the library imports.
      - For function calls, you don't need to import them, just include the function execution with the same parameters that were used.
    
    Here is an example of the expected output:
      ```python
      def pipeline_name(param1: Annotated[str, 'description of param1'] = default_value_param1, param2: Annotated[int, 'description of param2'] = default_value_param2) -> Dict[str, Any]:
          \"\"\"
          Pipeline description.
          Returns:
              Dict[str, Any]: description of the output.
          \"\"\"

          import necessary_libraries  # Dynamically extract required imports

          try:
              logging.info("Starting pipeline execution.")

              # Step 1: Call function A
              logging.info("Executing function_call_1")
              result_1 = function_call_1(param1)

              # Step 2: Execute Python code snippet
              logging.info("Executing extracted Python logic for something.")
              def some_python_logic(input_data):
                  \"\"\" Extracted Python code block. \"\"\"
                  # { Insert copied code blocks here }
                  return output_data
              
              result_2 = some_python_logic(result_1)

              # Step 3: Call function B
              logging.info("Executing function_B")
              final_result = function_B(result_2, param2)

              logging.info("Pipeline executed successfully.")
              return {"status": "success", "result": final_result}

          except Exception as e:
              logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
              return {"status": "error", "message": str(e)}
        ```
      """

CODE_WRITER_SYSTEM_PROMPT = '''code_writer. Write python codes to accomplish a given task.

General instructions:
    1. Write the full Python code solution in a single code block. Do not return incomplete codes or multiple code blocks.
    2. 'code_executor' will execute the code. If the output shows errors, correct them and present the complete, updated code.
    3. Wait until the 'code_executor' completes the execution and returns the output. If the output shows task completion, return a single 'TERMINATE'. Do not write or print it within the code block.
'''

PLOT_SYSTEM_PROMPT = CODE_WRITER_SYSTEM_PROMPT + '''
If you are asked to plot data, follow the guidance below:
    - Carefully follow the user's requirements of the expected figure.
    - Include clear and descriptive titles, axis labels, and legends in the figure.
    - Ensure the figure is visually appealing and easy to read by adjusting its styling elements such as font sizes, line widths, and colors.

Preferred Python libraries for plotting:
    - matplotlib, seaborn, plotly
'''

### Building Agents
user_proxy = autogen.ConversableAgent(
    name="Admin",
    system_message="A human admin to pose the task", 
    # code_execution_config={"use_docker": False},
    llm_config=False,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

# manager
manager = autogen.AssistantAgent(
    name="manager",
    system_message=MANAGER_SYSTEM_PROMPT,
    llm_config=gpt_config,
)

# coding agent
code_executor = autogen.UserProxyAgent(
    name="code_executor",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    system_message='''code_executor. Execute python code.''',
    # llm_config=gpt_config,
    code_execution_config= {
        "last_n_messages": "auto",
        "work_dir": WORK_DIR,
        "use_docker": False},
)

code_writer = autogen.AssistantAgent(
    name="code_writer",
    system_message=CODE_WRITER_SYSTEM_PROMPT,
    llm_config=gpt_config,
)

def coding(message: Annotated[str, "Describe the task to be solved."],
           file_path: Annotated[Optional[str], "(Optional) The path to an input csv file."]=None,
           ) -> str:   
    if file_path:
        message = f"{message}. The file to be analysed: {file_path}; it contains columns: {extract_column_name_from_csv(file_path)}."
    res = code_executor.initiate_chat(
        code_writer,
        clear_history=True,
        silent=False,
        message=message,
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Summarise all the necessary outputs needed to address the task. If the task involves saving files, make sure to return the saved file path."},
        max_turns=6,
    )
    return res.summary

# data visualiser
data_visualiser = autogen.AssistantAgent(
    name="data_visualiser",
    system_message=PLOT_SYSTEM_PROMPT,
    llm_config=gpt_config,
)

def plot_from_csv(message: Annotated[str, "The request of plotting data from a CSV file. e.g. 'Plot the mean of A against B.'"],
                  file_path: Annotated[str, "The path to the CSV file."],
                  save_path: Annotated[str, "The path to save the plot."]) -> str:
    column_names = extract_column_name_from_csv(file_path)
    res = code_executor.initiate_chat(
        data_visualiser,
        message=f'''{message}. The file to be analysed: "{file_path}", which contains column names: {column_names}. Save the plot to "{save_path}".''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt":"Summarise the outcome of the task, and return path of the saved plot."},
        max_turns=3,
    )
    return res.summary

# plot analyser
plot_analyser = MultimodalConversableAgent(name="plot_analyser",
                  system_message='''plot_analyser. You are a plant data scientist to analyse figures.
                  ''',
                  llm_config=gpt_config,)

def analyse_plot(message: Annotated[str, "The request of analysing a plot. e.g. 'what plant yields the most leaves at the end of the experiment?'"],
                 file_path: Annotated[str, "The path of the plot to be analysed, including the suffix."]) -> str:
    res = user_proxy.initiate_chat(
        recipient=plot_analyser,
        message=f'{message}. The plot to be analysed is <img {file_path}>.',
        max_turns=1,
    )
    return res.chat_history[-1]['content']

# table analyser
pdsllm = AzureOpenAI(
    api_token=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_API_URL'],
    api_version=os.environ['AZURE_API_VERSION'],
    deployment_name=os.environ['MODEL_NAME'],
)

@skill
def save_csv(df: pd.DataFrame,
             save_path: str) -> str:
    """
    Save a dataframe to a CSV file.
    Args:
        df (pd.DataFrame): The dataframe to save.
        save_path (str): The path to save the CSV file.
    """
    df.to_csv(save_path, index=False)
    return f"CSV file saved to {save_path}"

def compute_csv(
        message: Annotated[str, "The request of computing from a CSV file. Example: 'Compute the average of A for B on C.'"],
        file_path: Annotated[str, "The path of the CSV file to be analysed."],
        save_path: Optional[Annotated[str, "Optional: The path to save the results. If not provided, the result is returned as plain text."]] = None,
        ) -> str:
    df = pd.read_csv(file_path)
    # GPT setting
    agent_config = {"llm": pdsllm,}
    agent = Agent(df, config=agent_config,)
    # # Bamboo seeting
    # agent = Agent(df)

    if save_path:
        agent.add_skills(save_csv)
        res = agent.chat(f'{message}. The path to save results is: {save_path}.')
        exp = agent.explain()
        return f'The new csv is saved at {save_path}. Here is an explanation of the operation process: {exp}.'
    else:
        res = agent.chat(message)
        exp = agent.explain()
        return f"Here is the result: {res}. Explanation: {exp}."

def query_csv(message: Annotated[str, "Asking questions to a CSV file. Example: 'What is the maximum number of column A?'"], 
              file_path: Annotated[str, "The path of the CSV file to be analysed."],) -> str:
    df = pd.read_csv(file_path)
    
    # GPT setting
    agent_config = {"llm": pdsllm,}
    agent = Agent(df, config=agent_config,)

    # # Bamboo setting
    # agent = Agent(df)
    
    res = agent.chat(message)
    # if isinstance(res, np.integer):
    #     res = int(res)
    # if isinstance(res, np.floating):
    #     res = float(res)
    return str(res)

# RAG agent
retrieve_config = {
    "task": "qa", # default (basically qa + code), qa, or code
    "docs_path": ['./papers/phenotiki.pdf'],
    # "vector_db": "qdrant", # defualt is chroma
    "chunk_token_size": 2000,
    "collection_name": "knowledge_base", # default name is autogen-docs
    "get_or_create": True,
    "embedding_model": "all-mpnet-base-v2",  # we can also use openai's models here by defining an `embedding_function`
    "must_break_at_empty_line": True,
    "model": gpt_config['config_list'][0]["model"],
}

# retrieve info -> add info to prompts -> send to rag_assistant
rag_proxy = RetrieveUserProxyAgent(
    name="rag_proxy",
    # is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config=retrieve_config,
    code_execution_config=False,  # we don't want to execute code in this case.
    # description="You are a proxy to retrieve knowledge from documents and augment your prompts.",
)

# regular llm assistant
rag_assistant = RetrieveAssistantAgent(
    name="rag_assistant",
    system_message="You are a helpful assistant.",
    llm_config=gpt_config,
)

def retrieval_augmented_generation(problem: Annotated[str, "The question that can be answered based on knowledge retrieved from external source."]) -> str:
    '''
    Retrieve knowledge from the Phenotiki paper to answer a question. Provide the question as precise as possible.
    '''
    res = rag_proxy.initiate_chat(
        rag_assistant,
        message = rag_proxy.message_generator,
        problem = problem,
        silent=True,
    )
    return res.summary

# pipeline reproducer
pipeline_summariser = autogen.AssistantAgent(
    name="pipeline_summariser",
    system_message=PIPELINE_SUMMARISER_SYSTEM_PROMPT,
    llm_config=gpt_config,
)

def extract_pipeline(
    pipeline_name: Annotated[str, "The name of the pipeline to be saved."],
    chat_log_path: Annotated[str, "The path to the chat log file."] = './autogen_logs/runtime.log',
) -> str:
    """
    Extract executed function calls and code to form a reproducible pipeline from a chat log.
    """
    # Load chat log
    log_data = load_chat_log(chat_log_path)
    summary_method = "last_msg" # or "reflection_with_llm"
    
    res = user_proxy.initiate_chat(
        pipeline_summariser,
        clear_history=True,
        silent=False,
        message=f'''Extract a pipeline from the chat log. Name the extracted pipeline as {pipeline_name}. Users should be able to use new data to run the pipeline.
        Here is the chat_log {log_data}.''',
        summary_method=summary_method,
        # summary_args={"summary_prompt": "Return only the extracted Python code block as it is. Ignore any irrelevant command like 'python' or execute command."},
        max_turns=1,
    )
    pipeline_code = res.summary
    save_pipeline(pipeline_name, pipeline_code)
    return f"Pipeline {pipeline_name} extracted successfully."

register_function(
    perform_anova,
    caller=manager,
    executor=user_proxy,
    name="perform_anova",
    description="Perform Mixed-design Repeated Measures ANOVA on given data (Greenhouse-Geisser correction will be automatically applied if needed)",
)

register_function(
    perform_tukey_test,
    caller=manager,
    executor=user_proxy,
    name="perform_tukey_test",
    description="Perform Post-hoc Tukey-Kramer test on given data",
)

register_function(
    extract_pipeline,
    caller=manager,
    executor=user_proxy,
    name="extract_pipeline",
    description="Extract and save a reproducible pipeline from chat history. Ask user to provide a name for the pipeline.",
)

register_function(
    get_pipeline_zoo,
    caller=manager,
    executor=user_proxy,
    name="get_pipeline_zoo",
    description="Get the information of all registered pipelines. It is useful when a user wants to know what pipelines are available before executing any.",
)

register_function(
    get_pipeline_info,
    caller=manager,
    executor=user_proxy,
    name="get_pipeline_info",
    description="Get the information of a specific pipeline. This is useful for you to know how to use a pipeline selected by the user, including the description, arguments, and output type.",
)

register_function(
    execute_pipeline,
    caller=manager,
    executor=user_proxy,
    name="execute_pipeline",
    description="Execute a saved pipeline from the pipeline zoo. Before executing a pipeline, you must call 'get_pipeline_zoo' to know what pipelines are available, and call 'get_pipeline_info' to understand how to use the selected pipeline.",
)

register_function(
    calculator,
    caller=manager,
    executor=user_proxy,
    name="calculator",
    description="Perform basic arithmetic operations between two integers.",
)

register_function(
    search_and_scrape,
    caller=manager,
    executor=user_proxy,
    name="google_search",
    description="Search and scrape content from the web. Results are returned in a dictionary. Useful when you need to find information on a specific topic.",
)

register_function(
    get_model_zoo,
    caller=manager,
    executor=user_proxy,
    name="get_model_zoo",
    description="Check available computer vision checkpoints. Must be called before using computer vision models.",
)

register_function(
    infer_instance_segmentation,
    caller=manager,
    executor=user_proxy,
    name="infer_instance_segmentation",
    description="Perform instance segmentation on plant images",
)

register_function(
    infer_image_classification,
    caller=manager,
    executor=user_proxy,
    name="infer_image_classification",
    description="Perform image classification on plant images",
)

register_function(
    infer_image_regression,
    caller=manager,
    executor=user_proxy,
    name="infer_image_regression",
    description="Perform image regression on plant images",
)
register_function(
    infer_image_classification_dinov2,
    caller=manager,
    executor=user_proxy,
    name="infer_image_classification_dinov2",
    description="Perform nutrient deficiency classification using DINOv2 ViT-S/14 full fine-tuned models contributed by Narendren S V (IIT Bombay). IMPORTANT - image requirements differ per crop: rice requires close-up RGB leaf images (4-class nitrogen deficiency severity: mild/moderate/severe/very_severe); wheat requires UAV top-down aerial field images ONLY and NOT close-up leaf images (5-class: healthy/nitrogen/phosphorus/potassium/severe deficiency); maize requires close-up RGB leaf or plant images (6-class: healthy/nitrogen/phosphorus/potassium/zinc/severe deficiency). Checkpoint format: Naren1704/{crop}_nutrient-deficiency_rgbdataset_dinov2_fullft",
)

register_function(
    compute_phenotypes_from_ins_seg,
    caller=manager,
    executor=user_proxy,
    name="compute_phenotypes_from_ins_seg",
    description="Compute phenotypes from an instance segmentation result file",
)

register_function(
    coding,
    caller=manager,
    executor=user_proxy,
    name="coding",
    description="Write and execute code to solve tasks. Please provide a complete task description rather than concrete code as the input to this function.",)

register_function(
    analyse_plot,
    caller=manager,
    executor=user_proxy,
    name="analyse_plot",
    description="Analyse a plot using GPT-4o.",
)

register_function(
    compute_csv,
    caller=manager,
    executor=user_proxy,
    name="compute_from_csv",
    description="Compute statistics or new values from a CSV file. Optionally, it saves the results to a new file.",
)

register_function(
    query_csv,
    caller=manager,
    executor=user_proxy,
    name="query_csv",
    description="Ask a question to a CSV file such as which image has the most leaf count. It does not generate a new file.",
)

register_function(
    plot_from_csv,
    caller=manager,
    executor=user_proxy,
    name="plot_from_csv",
    description="Plot data from a CSV file. Be sure to provide details of requirements and the path to save the plot.",
)

register_function(
    retrieval_augmented_generation,
    caller=manager,
    executor=user_proxy,
    name="RAG",
    description="Retrieve knowledge from the Phenotiki paper. Use this only to retrieve information (e.g. asking questions starting with what/how/...). You need to reason the retrieved information to solve the task.",
)

# Model finetuning function
register_function(
    get_dataset_format,
    caller=manager,
    executor=user_proxy,
    name="get_dataset_format",
    description="Instruct the user to prepare a dataset in the required format to train a model.",
)

register_function(
    prepare_dataset,
    caller=manager,
    executor=user_proxy,
    name="prepare_dataset",
    description="When the user upload a dataset for model training, use this function to process the dataset into the required format.",
)

register_function(
    finetune_image_classification,
    caller=manager,
    executor=user_proxy,
    name="finetune_image_classification",
    description="Train an image classification model on a user uploaded dataset.",
)

register_function(
    finetune_image_regression,
    caller=manager,
    executor=user_proxy,
    name="finetune_image_regression",
    description="Train an image regression model on a user uploaded dataset.",
)

register_function(
    finetune_instance_segmentation,
    caller=manager,
    executor=user_proxy,
    name="finetune_instance_segmentation",
    description="Train an instance segmentation model on a user uploaded dataset.",
)

register_function(
    make_dir,
    caller=manager,
    executor=user_proxy,
    name="make_dir",
    description="Check if a directory exists, and create it if it does not. Call it whenever you need to save files to a directory.",
)

## adding new tools starts
try:
    import warnings
    from utils.registry import auto_discover_tools, register_all_tools
    auto_discover_tools()
    register_all_tools(manager, user_proxy)
    warnings.warn("[PhenoAssistant] User tools auto-registered from ./functions", RuntimeWarning)
except Exception as _e:
    warnings.warn(f"[PhenoAssistant] Skipped tool auto-registration: {_e}", RuntimeWarning)

print("PhenoAssistant's available tools:")
for i, tool in enumerate(manager.llm_config["tools"]):
    print(f"Tool {i+1}: Name: {tool['function']['name']}, Description: {tool['function']['description']}")
## adding new tools ends

# ── Contribution 2: Model Selector Agent ─────────────────────────────────────
from utils.model_selector import select_model as _select_model

def get_best_model(user_query: str) -> str:
    """
    Select the single best vision model for a plant phenotyping task.
    Uses a 3-question decision tree instead of showing all models at once.
    Returns the exact model ID to pass to infer_* functions.
    """
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_URL"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    return _select_model(user_query, client, os.environ["MODEL_NAME"])

register_function(
    get_best_model,
    caller=manager,
    executor=user_proxy,
    name="get_best_model",
    description=(
        "Select the single best vision model checkpoint for a plant phenotyping task. "
        "Use this instead of get_model_zoo when the user wants to run inference. "
        "Returns the exact model ID to pass to infer_instance_segmentation, "
        "infer_image_classification, or infer_image_regression."
    ),
)

print("[Contribution 2] get_best_model registered successfully.")


# ── Contribution 2: Tool Selector Agent ──────────────────────────────────────
from utils.tool_selector import ToolSelectorIndex

tool_index = ToolSelectorIndex(manager, model_name="all-mpnet-base-v2")

def start_task(user_message: str, k: int = 7, **kwargs):
    """
    Use this instead of user_proxy.initiate_chat() in notebooks.
    Filters the tool list to top-k before the manager sees it.
    """
    with tool_index.patch_manager(user_message, k=k):
        return user_proxy.initiate_chat(manager, message=user_message, **kwargs)

print("[Contribution 2] ToolSelectorIndex ready. Use start_task() in notebooks.")
