"""
Run this example to see the sketch for a RouteNotDiamond node.

```bash
pyenv virtualenv 3.11 substrate-nd-example
pyenv activate substrate-nd-example
pip install substrate
python nd_example.py
```
"""
import os

from dotenv import load_dotenv

from substrate import Substrate, ComputeText, sb, Secrets, If
from substrate.run_notdiamond import RouteNotDiamond

load_dotenv()

SUBSTRATE_API_KEY = os.getenv("SUBSTRATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def quickstart():
    substrate = Substrate(
        api_key=SUBSTRATE_API_KEY,
        secrets=Secrets(
            openai=OPENAI_API_KEY,
            anthropic=ANTHROPIC_API_KEY,
        ),
    )

    story = ComputeText(
        prompt="Write a short story about a raccoon eating a banana.",
    )
    summary = ComputeText(
        prompt=sb.format("summarize: {story}", story=story.future.text)
    )
    response = substrate.run(summary)
    summary_out = response.get(summary)
    print(response)
    print(summary_out)


def route_quickstart():
    substrate = Substrate(
        api_key=SUBSTRATE_API_KEY,
    )

    story = ComputeText(
        prompt="Write a short story about a raccoon eating a banana.",
    )
    route = RouteNotDiamond(
        route_input=story,
        models=["gpt-4o", "claude-3-5-sonnet-20240620"],
    )
    summarize_4o = ComputeText(
        prompt=sb.concat(
            "summarize: ",
            story.future.text,
        ),
        model="gpt-4o",
    )
    summarize_sonnet = ComputeText(
        prompt=sb.concat(
            "summarize: ",
            story.future.text,
        ),
        model="claude-3-5-sonnet-20240620",
    )
    summary = If(
        condition=sb.jq(route.future.text, "output == 'gpt-4o'"),
        value_if_true=summarize_4o.future.text,
        value_if_false=summarize_sonnet.future.text,
    )

    print(f"story={story.id}, summarize_4o={summarize_4o.id}, summarize_sonnet={summarize_sonnet.id}, summary={summary.id}")

    # print(substrate.visualize(summary))
    # response = substrate.run(summary)
    # print(response)
    response = substrate.run(route)
    print(response)

if __name__ == "__main__":
    route_quickstart()
