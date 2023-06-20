# Overview

This repository is dedicated to the systematic development of an incrementally more sophisticated artificial intelligence (AI) system. The central goal is a continual, iterative improvement in system capabilities, through the adoption of a human-in-the-loop strategy, with advancements measured against a rigorous set of performance metrics.

The developmental framework relies heavily on a human-in-the-loop model. In the early stages, human experts will be responsible for the bulk of the work, performing most tasks and providing decision-making input. As the AI system is trained and improves, through the utilization of machine learning algorithms and reinforcement learning techniques, it will progressively undertake a larger share of the workload. The ultimate aim is for the AI to achieve the capacity to perform an expanding range of tasks with decreasing human intervention.

The key operational document in this repository is 'plan.md', a detailed blueprint outlining the steps for enhancing the system's intelligence. This document defines tasks, outlines procedures, and provides justifications, and will be frequently updated by human collaborators. The tasks specified will be decomposed to the granularity that the current state of the AI system can handle. 

Meanwhile, a suite of tools will continuously monitor 'plan.md', identifying and tracking newly specified tasks, and providing feedback to both the AI system and the human collaborators. This monitoring serves as a critical mechanism for maintaining progress, identifying issues, and ensuring that the work stays aligned with the overarching strategic goals.

This human-in-the-loop approach drives the process of iterative improvement in the AI system, fostering an environment of continuous learning and adaptation. This project represents a concerted effort to push the boundaries of what is achievable in the realm of system intelligence and to explore new frontiers in the integration of human and machine learning.
You are a key component of a system designed to assist users in managing complex projects. Your roles involve clarifying user intent to prevent assumptions and executing commands on the user's terminal.

## Interaction Structure

Your interactions are structured around two types of messages:

### Messages you receive:

These include 'prompt', 'feedback', 'previous', 'answers', and 'outputs'.

- `prompt`: This field contains guiding instructions for task execution.
- `feedback`: This field incorporates user suggestions or criticisms post-task execution.
- `answers`: This field is an array of objects with question-answer pairs provided by the user. Each object follows the structure: {question: '...', answer: '...'}.
- `outputs`: This field is an array of objects containing command and corresponding output pairs. Each object follows the structure: {command: '...', output: '...'}.
- `previous`: This is an array of previous input messages that the AI has received.

### Messages you respond with:

These include 'context', 'goal', 'questions', and 'commands'.

- `context`: This field summarizes the outcomes from prior actions, encapsulating your current understanding of the situation.
- `goal`: Given the 'context', this field outlines the objective you plan to achieve next.
- `questions`: An array of questions to present to the user to ensure that you correctly interpret the user's intent.
- `commands`: These are shell commands planned to achieve the 'goals'. If the task involves writing code, Golang should be the preferred language, unless specified otherwise.

## Example Interaction
### Example message you might receive
```json
{
  "prompt": "Please execute the uptime command.",
  "feedback": "",
  "previous": [],
  "answers": [{ "question": "Do you want the uptime displayed in a specific format?", "answer": "No, default is fine." }],
  "outputs": [{ "command": "uptime", "output": "up 10 days, 20:00" }]
}
```

### Example message you might respond
```json
{
  "context": "The user requested the operational duration of the current machine and prefers the default format.",
  "goal": "To provide the uptime information to the user.",
  "commands": ["uptime"],
  "questions": []
}
```

### Command Recomentations

When you intend to modify local files, use the `nl` command on the targeted file to obtain line numbers:

```bash
nl targetfile.txt
```

After identifying the line numbers for modification, write a patch file. Here's an example of how to use `echo` to create the patch file:

```bash
echo '--- targetfile.txt' > change.patch
echo '+++ targetfile.txt' >> change.patch
echo '@@ -1,1 +1,1 @@' >> change.patch
echo '-original line' >> change.patch
echo '+modified line' >> change.patch
```

The above commands create a patch file named `change.patch`. The line starting with '@@' indicates that we are changing line 1 in the original file to line 1 in the new file. The '-' denotes the original line content, while the '+' denotes the new line content.

Apply the changes to the target file using `patch`:

```bash
patch -p1 < change.patch
```

This method allows precise and efficient file modifications.

## Task Management
Task Management is integral to the organization and successful completion of any project. In this system, tasks are meticulously defined, linked to their specifications in a document called `plan.md`, and managed through a dedicated program, `task`.

### Task specifications

Tasks are defined by a structured format with distinct fields, ensuring each task has complete, clearly defined specifications. Below is an example of a task specification:


```task
{
    id: 0,
    name: "example task",
    status: "completed",
    anchor: "###task-specifications",
    blockers: []
}
```

Fields in Task Specification
1. id: A unique, incrementing integer identifying each task. No two tasks will have the same ID.
2. name: A concise, descriptive label for the task, summarizing its purpose in less than 80 characters.
3. status: Indicates the current state of the task. It can take one of three values:
- queued: The task is scheduled but has not yet started.
- in-progress: The task is currently underway.
- completed: The task has been successfully finished.
4. anchor: A reference to the specific section of the plan.md document that provides detailed task specifications. Each anchor must be unique to unequivocally identify each task.
5. blockers: A list of task IDs that must be completed before the current task can proceed. This helps manage task dependencies and ensures tasks are completed in the correct sequence.

The plan.md document serves as a complete blueprint of the project, with detailed descriptions and specifications for all tasks. To keep the plan organized, tasks are embedded into the respective sections of the plan.md where they belong.

# The task program
The task program facilitates managing and tracking tasks according to their specifications. It allows retrieving any queued task, updating task status, and handling other related operations.

Here are the key commands provided by the task program:

1. task get [id]: Retrieves the task with the given id.
2. task status [id] [new_status]: Changes the status of the task with the given id to the new status.
3. task blockers [id] [id1, id2, ...]: Sets the tasks (id1, id2, ...) as blockers for the task with the given id.
4. task queue: Retrieves the list of all tasks that are currently queued.
5. task progress [id]: Marks the task with the given id as in progress.
6. task complete [id]: Marks the task with the given id as completed.
Adhering to these specifications and using the task program ensures tasks are systematically managed, tracked, and updated, facilitating clear communication and efficient workflow within the project.

# Roadmap

## Task: Plan for Markdown Linting System

**Task Specifications:**

```task
{
    "id": 1,
    "name": "Plan for Markdown Linting System",
    "status": "queued",
    "anchor": "#task-plan-for-markdown-linting-system",
    "blockers": []
}
```

**Task Description:**

The first task on our agenda is to create a comprehensive plan for a Markdown linting system. This system will ensure that all tasks specified in the 'plan.md' file adhere to the defined task structure and guidelines. Markdown linting will enhance the overall quality and consistency of the project documentation.

**Task Goals:**

- Develop a detailed plan for the Markdown linting system.
- Specify the rules and requirements for valid task specifications.
- Define the expected format and guidelines for task descriptions.
- Establish a mechanism for detecting and reporting any non-compliant tasks.

**Task Deliverables:**

- A well-documented plan outlining the implementation details of the Markdown linting system.
- Clearly defined rules and guidelines for task specifications.
- Validation mechanisms to verify the correctness and compliance of all tasks.
- Append the plan to this section of the document

### Plan
- Define the Purpose: Enforce a consistent style and structure for the plan.md document, ensuring all tasks follow the defined specification.
- Key Components: A Markdown parser, a set of custom linting rules, a linter to apply these rules, and a reporter to provide feedback.
- Implementation Approach: Parse the plan.md document into an AST using the Markdown parser. Then, traverse the AST and apply the linting rules to each task.
- Specify Rules and Requirements: The rules for valid task specifications will be based on the existing task structure defined in plan.md.
- Define Task Descriptions Format and Guidelines: Task descriptions should be clear, concise, and contain sufficient information to understand the tasks objectives.
- Non-compliant Tasks Detection and Reporting: The linting system should provide clear and concise error messages when a task does not comply with the specifications.
- Testing and Validation Process: Comprehensive test cases covering various scenarios will be developed to verify the correctness and compliance of the linting system.

## Task: Plan to extract a dataset from chat


### Prompt

```
Your task is to parse a sequence of JSON files representing a conversation about coding tasks between a user and another language model. Each JSON file in the sequence is named either 'user_X.json' or 'llm_X.json', where X is a number indicating the order of the conversation. These files are located in a folder named 'chat'.

You need to perform the following steps:

1. Extract the initial task description from the first file in each sequence, 'user_0.json'.
   
2. Find the final satisfactory code snippet. This is in the last 'llm_X.json' file of each sequence where the user has agreed that the solution is satisfactory.

3. Create an input-output pair for each sequence. The initial task description is the input, and the satisfactory code snippet is the output.

Your output should be a JSON file for each sequence. Each JSON file should have the following structure:

    {
        "prompt": "the initial task description",
        "completion": "the satisfactory code snippet"
    }

Save these JSON files in a new folder named 'finetune'. Name each file as 'user_10-llm_15.json', where 10 and 15 represent the start and end of the sequence that has been processed."

This should provide a clear and precise task definition for the language model.
```
