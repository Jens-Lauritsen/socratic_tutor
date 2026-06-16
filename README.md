# SocraticCode Tutor

> **"AI should make developers better, not just faster."**

An open-source AI programming tutor that teaches using the **Socratic method** — asking questions, providing progressive hints, and evaluating understanding rather than handing out solutions.

## Why SocraticCode?

Modern AI coding assistants often encourage:
- Copy/pasting solutions without understanding
- Shallow knowledge of programming concepts
- Dependency on AI for every problem
- Skipping debugging and reasoning

**SocraticCode solves this** by acting like a patient mentor who:
- Asks questions before giving answers
- Provides hints progressively (4 levels)
- Encourages you to explain your thinking
- Adapts to your skill level
- Evaluates whether you actually understand

## Features

### 1. Socratic Dialogue
Instead of immediately answering "Why is my function failing?", the tutor asks:
- *"What do you think might be causing this?"*
- *"Where would you start debugging?"*
- *"What does this error usually indicate?"*

### 2. Progressive Hint System (4 Levels)
| Level | Type | Example |
|-------|------|---------|
| 1 | Question | "What does a 401 status code indicate?" |
| 2 | Direction | "Look at how authentication tokens are created." |
| 3 | Concept | "This relates to JWT token validation flow." |
| 4 | Partial Solution | "Here's how you might structure the auth middleware..." |

### 3. Learning Profile (Local & Private)
```json
{
  "skills": {
    "variables": 0.9,
    "functions": 0.8,
    "classes": 0.4,
    "testing": 0.3
  },
  "common_mistakes": [
    "poor error handling",
    "large functions"
  ]
}
```
Stored locally in `~/.config/socratic-code/profile.json` — no cloud, no tracking.

### 4. Exercise Mode
Generates personalized programming exercises based on your skill profile:
- Chooses appropriate difficulty
- Provides progressive hints
- Evaluates your solution
- Explains mistakes
- Suggests next concepts to learn

### 5. Understanding Evaluation
After you solve a problem, the tutor asks: *"Explain why your solution works."*
Your explanation is evaluated for depth of understanding, not just correctness.

## Installation

```bash
pip install socratic-code
```

For local development:

```bash
git clone https://github.com/your-org/socratic-code.git
cd socratic-code
pip install -e ".[dev]"
```

## Quick Start

### 1. Set your API key

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# or use a local model with Ollama
export SOCRATIC_LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Start a session

```bash
socratic start --topic "Python classes"
```

### 3. Generate an exercise

```bash
socratic exercise "loops and functions"
```

### 4. View your progress

```bash
socratic progress
```

## Usage Examples

### Interactive Session

```
$ socratic start

What are you learning today?
> Python error handling

Welcome! I'm your Socratic programming tutor.
I'll guide you with questions and hints rather than giving direct answers.

You: My try/except block isn't catching the error

Tutor: What type of error are you expecting, and what type does
       your except clause actually catch?

You: I'm not sure...

Tutor: Try adding `except Exception as e: print(type(e).__name__)`
       before your specific handler. What types do you see?
```

### Commands During a Session

| Command | Description |
|---------|-------------|
| `/hint [1-4]` | Request a hint at a specific level |
| `/exercise` | Generate a practice exercise |
| `/evaluate` | Explain what you learned for evaluation |
| `/progress` | View your learning profile |
| `/quit` | End the session |

## Architecture

```
                 User
                  |
                  v
          Socratic Tutor Engine
                  |
       +----------+----------+
       |          |          |
       v          v          v
 Context     Teaching    Evaluation
 Agent       Agent       Agent
                  |
                  v
            Learning Profile
                  |
                  v
              LLM Provider
         (OpenAI / Anthropic / Ollama)
```

### Components

- **Tutor Engine** — conversation flow, hint progression, session management
- **Teaching Agent** — chooses next hint, adapts teaching style
- **Context Agent** — understands user code, finds relevant files
- **Evaluation Agent** — checks understanding, scores explanations
- **Learning Profile** — local JSON storage of skills and progress

## Supported LLM Providers

| Provider | Environment Variable | Default Model |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| Ollama | `OLLAMA_BASE_URL` | `llama3.2` |

Set `SOCRATIC_LLM_PROVIDER` to choose: `openai`, `anthropic`, or `ollama`.

Users provide their own API keys — **the project does NOT pay for AI usage**.

## Development

### Project Structure

```
socratic-code/
├── src/
│   └── socratic/
│       ├── __init__.py
│       ├── cli.py            # CLI entry point
│       ├── tutor.py          # High-level API
│       ├── engine.py         # Conversation engine
│       ├── prompts.py        # Socratic prompt templates
│       ├── models.py         # Pydantic data models
│       ├── progress.py       # Learning profile storage
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── context.py    # Code context analysis
│       │   ├── teaching.py   # Hint progression logic
│       │   └── evaluation.py # Understanding assessment
│       └── providers/
│           ├── __init__.py   # LLM provider abstraction
│           └── (openai, anthropic, ollama implementations)
├── tests/
│   └── test_core.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=socratic --cov-report=term-missing
```

### Design Principles

- **Educational value over code generation** — The AI should teach, not replace thinking
- **Transparency over magic** — All prompts are visible and customizable
- **Modular architecture** — Each agent has a single responsibility
- **User privacy** — All data stored locally, no cloud dependency
- **Open-source friendly** — MIT licensed, easy to contribute

## Roadmap

### MVP (Current)
- [x] CLI application
- [x] Local user progress tracking
- [x] Socratic conversation engine
- [x] Multiple LLM providers (OpenAI, Anthropic, Ollama)
- [x] Exercise generation
- [x] Understanding evaluation

### Future
- [ ] VS Code extension
- [ ] Web frontend (Next.js)
- [ ] Multi-language support
- [ ] Code execution sandbox
- [ ] Peer learning groups
- [ ] Curriculum/path generation
- [ ] Spaced repetition for concepts

## Contributing

Contributions are welcome! See our design principles above — prioritize educational value, transparency, and modularity.

## License

MIT — see [LICENSE](LICENSE) for details.

---

*"The unexamined code is not worth writing."* — adapted from Socrates
